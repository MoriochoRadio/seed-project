"""
pipeline.py  ·  AI-CASA 통합 파이프라인 (Phase 1-3)
──────────────────────────────────────────────────
사용법:
    from src.pipeline import SpermAnalysisPipeline
    pipeline = SpermAnalysisPipeline()
    result   = pipeline.analyze('video.mp4')
    pipeline.print_report(result)
"""

import os
import cv2
import numpy as np

from .detector    import SpermDetector
from .tracker     import SpermTracker
from .analyzer    import MotilityAnalyzer
from .interpreter import (interpret_motility, assess_confidence,
                           interpret_kinematics, get_overall_judgment)


class SpermAnalysisPipeline:
    """AI-CASA 통합 파이프라인 (운동성 + 키네마틱 + 형태)"""

    def __init__(self,
                 yolo_path    : str = None,
                 model_path   : str = None,
                 tracker_config: str = None,
                 morph_path   : str = None):

        base = r'C:\Users\neo62\sperm-ai'

        yolo_path      = yolo_path or os.path.join(
            base, 'models', 'yolo11_sperm_v2',
            'weights', 'best.pt')
        model_path     = model_path or os.path.join(
            base, 'models', 'motility_ensemble.pkl')
        tracker_config = tracker_config or os.path.join(
            base, 'bytetrack_custom.yaml')
        morph_path     = morph_path or os.path.join(
            base, 'models', 'morphology_efficientnet_b3_v3.pt')

        self.detector  = SpermDetector(yolo_path)
        self.tracker   = SpermTracker(self.detector, tracker_config)
        self.analyzer  = MotilityAnalyzer(model_path)

        try:
            from .morphology import MorphologyAnalyzer
            self.morph_analyzer = MorphologyAnalyzer(morph_path)
        except Exception:
            self.morph_analyzer = None
            print("⚠️  형태 분석 모델 없음 → 운동성 분석만 수행")

    # ── 정자 크롭 추출 ─────────────────────────────────────
    def _extract_crops(self, video_path: str,
                       max_frames: int = 50) -> list:
        """영상에서 YOLO11 탐지 기반 정자 크롭 추출"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // max_frames)
        crops = []

        for fidx in range(0, min(total, max_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.model(
                frame, verbose=False, conf=0.3)
            if results[0].boxes is None:
                continue

            H, W = frame.shape[:2]
            for box, cls in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy().astype(int)
            ):
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w < 15 or h < 15:
                    continue
                pad = max(w, h) // 2
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(W, x2 + pad)
                y2p = min(H, y2 + pad)
                crop = frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crops.append(clahe.apply(gray))

        cap.release()
        return crops

    # ── 전체 분석 ──────────────────────────────────────────
    def analyze(self, video_path: str,
                verbose: bool = True) -> dict:
        """
        영상 입력 → 운동성 + 키네마틱 + 형태 통합 분석
        """
        name = video_path.split('\\')[-1]
        if verbose:
            print(f"분석 시작: {name}")
            print("─" * 50)

        # Step 1: 정자 수 추정
        N, counts = self.detector.estimate_sperm_count(video_path)
        if verbose:
            print(f"[1/5] 정자 수 기준값: {N}개")

        # Step 2: 추적
        track_history = self.tracker.track_video(video_path)
        if verbose:
            print(f"[2/5] 추적 완료: {len(track_history)}개 ID")

        # Step 3: 특징 추출
        features = self.tracker.extract_features(track_history)
        if features is None:
            return None
        if verbose:
            print(f"[3/5] 특징 추출 완료")

        # Step 4: 키네마틱 계산
        kinematics = self.tracker.compute_sample_kinematics(
            track_history, fps=50.0)

        # Step 5: 운동성 예측
        motility = self.analyzer.predict(features)
        if verbose:
            print(f"[4/5] 운동성 분석 완료")

        # Step 6: 형태 분석
        morphology        = {}
        morphology_interp = {}
        if self.morph_analyzer is not None:
            crops = self._extract_crops(video_path)
            if crops:
                morphology = self.morph_analyzer.analyze_crops(
                    crops)
                morphology_interp = \
                    self.morph_analyzer.interpret_morphology(
                        morphology)
            if verbose:
                print(f"[5/5] 형태 분석 완료 "
                      f"(크롭 {len(crops)}개)")

        # Step 7: WHO 해석 + 신뢰도
        interp  = interpret_motility(
            motility['progressive'],
            motility['non_progressive'],
            motility['immotile'])
        quality = assess_confidence(counts, track_history, N)

        # Step 8: 종합 판정
        morph_level = morphology_interp.get('level', '')
        overall     = get_overall_judgment(
            interp['level'],
            morph_level,
            quality['confidence_score'])

        return {
            'video':              name,
            'N':                  N,
            **motility,
            **interp,
            **quality,
            'kinematics':         kinematics,
            'morphology':         morphology,
            'morphology_interp':  morphology_interp,
            **overall,
        }

    # ── 보고서 출력 ────────────────────────────────────────
    @staticmethod
    def print_report(result: dict,
                     participant_id: str = None) -> None:
        """AI-CASA 통합 보고서 출력 (일반인 친화적)"""

        W = 60

        title = "AI 정자 분석 보고서"
        if participant_id:
            title += f"  (참가자 {participant_id})"

        # ── 헤더 ──────────────────────────────────────────
        print(f"\n{'='*W}")
        print(f"  🔬 {title}")
        print(f"{'='*W}")

        # ── 분석 품질 ─────────────────────────────────────
        score = result['confidence_score']
        icon  = result['confidence_icon']
        level = result['confidence_level']

        print(f"\n【 분석 품질 】")
        print(f"  {icon} 신뢰도: {level} ({score}점 / 100점)")

        if result['quality_issues']:
            for issue in result['quality_issues']:
                print(f"  ⚠️  {issue}")
        if result['needs_retake']:
            print(f"  🔄 재촬영을 권고합니다")
            print(f"     (신뢰도가 낮아 결과가 부정확할 수 있습니다)")

        # ── 운동성 수치 ───────────────────────────────────
        print(f"\n{'─'*W}")
        print(f"【 정자 운동성 분석 】  (탐지된 정자: {result['N']}개)")
        print(f"{'─'*W}")

        prog     = result['progressive']
        non_prog = result['non_progressive']
        immotile = result['immotile']
        total    = result['total_motile']

        def bar(val, color='█'):
            return color * int(val / 5)

        print(f"\n  앞으로 잘 나아가는 정자  "
              f"(전진 운동성):  {prog:>5.1f}%  {bar(prog)}")
        print(f"  조금 움직이는 정자      "
              f"(비전진 운동성):{non_prog:>5.1f}%  {bar(non_prog, '░')}")
        print(f"  움직이지 않는 정자      "
              f"(비운동성):     {immotile:>5.1f}%  {bar(immotile, '·')}")
        print(f"  {'─'*45}")
        print(f"  전체 움직이는 정자      "
              f"(총 운동성):    {total:>5.1f}%")

        # ── WHO 기준 해석 ─────────────────────────────────
        print(f"\n  📋 WHO 기준 검토 결과")
        for line in result['interpretation']:
            print(f"     {line}")

        # ── 키네마틱 ──────────────────────────────────────
        if result.get('kinematics'):
            k    = result['kinematics']
            kint = interpret_kinematics(k)

            print(f"\n{'─'*W}")
            print(f"【 정자 운동 패턴 분석 (CASA 키네마틱) 】")
            print(f"{'─'*W}")
            print(f"\n  속도 지표:")
            print(f"  {'곡선 속도 (VCL)':<22} {k['VCL_mean']:>6.1f} µm/s"
                  f"  (기준 ≥25 µm/s)")
            print(f"  {'직선 속도 (VSL)':<22} {k['VSL_mean']:>6.1f} µm/s"
                  f"  (기준 ≥15 µm/s)")
            print(f"  {'평균경로 속도 (VAP)':<22} {k['VAP_mean']:>6.1f} µm/s"
                  f"  (기준 ≥20 µm/s)")

            print(f"\n  운동 패턴 지표:")
            print(f"  {'직진성 (STR)':<22} {k['STR_mean']:>6.3f}"
                  f"       (기준 ≥0.80,  1에 가까울수록 직선)")
            print(f"  {'직선성 (LIN)':<22} {k['LIN_mean']:>6.3f}"
                  f"       (기준 ≥0.50)")
            print(f"  {'진동도 (WOB)':<22} {k['WOB_mean']:>6.3f}"
                  f"       (기준 ≥0.70)")

            if kint:
                print(f"\n  🔍 해석:")
                for line in kint['interpretations']:
                    print(f"     {line}")
                print(f"  📌 {kint['ppms_note']}")

        # ── 형태 분석 ─────────────────────────────────────
        if result.get('morphology') and result['morphology']:
            m    = result['morphology']
            mint = result.get('morphology_interp', {})

            print(f"\n{'─'*W}")
            print(f"【 정자 형태 분석 】")
            print(f"{'─'*W}")
            print(f"\n  분석된 정자: {m['n_analyzed']}개  |  "
                  f"정상 형태: {m['n_normal']}개 "
                  f"({m['normal_rate']:.1f}%)")
            print(f"  ※ 참고용 수치 (도메인 차이로 과대 추정 가능)")

            print(f"\n  부위별 이상 비율:")
            part_names = {
                'head':     '머리    (두부)',
                'acrosome': '첨체    (머리 앞부분)',
                'vacuole':  '공포    (머리 내 공간)',
                'tail':     '꼬리    (미부)',
            }
            for p, name in part_names.items():
                rate    = m['abnormal_rates'][p]
                flag    = '⚠️ ' if rate >= 50 else '✅'
                bar_str = '█' * int(rate / 5)
                print(f"  {flag} {name:<22} 이상 {rate:>5.1f}%  "
                      f"{bar_str}")

            status = mint.get('status', '-')
            s_icons = {'정상 범위': '✅', '경계 범위': '⚠️ ', '주의 필요': '🔴'}
            s_icon  = s_icons.get(status, '⚪')
            print(f"\n  {s_icon} 정상 형태율: "
                  f"{m['normal_rate']:.1f}%  →  {status}")
            print(f"     (WHO 기준: 4% 이상이면 정상)")

        # ── 종합 판정 ─────────────────────────────────────
        print(f"\n{'='*W}")
        print(f"【 종합 판정 】")
        print(f"{'='*W}")

        o_icon   = result.get('overall_icon', '⚪')
        o_status = result.get('overall_status', '-')
        o_msg    = result.get('overall_message', '')
        m_status = result['status']

        mot_icons = {'정상 범위': '✅', '경계 범위': '⚠️ ', '주의 필요': '🔴'}
        mot_icon  = mot_icons.get(m_status, '⚪')
        print(f"\n  운동성 판정:  {mot_icon} {m_status}")

        if result.get('morphology'):
            m_s     = result.get('morphology_interp', {}).get('status', '-')
            m_icons = {'정상 범위': '✅', '경계 범위': '⚠️ ', '주의 필요': '🔴'}
            m_i     = m_icons.get(m_s, '⚪')
            print(f"  형태 판정:    {m_i} {m_s}  ※참고용")

        print(f"\n  {o_icon} 최종 판정: [{o_status}]")
        print(f"\n  💬 권고사항:")
        print(f"   {o_msg}")

        # ── 안내 문구 ─────────────────────────────────────
        print(f"\n{'─'*W}")
        print(f"  ⚠️  주의사항")
        print(f"  이 결과는 AI 보조 분석이며 의학적 진단이 아닙니다.")
        print(f"  정확한 진단은 반드시 전문 의료기관에서 받으시기 바랍니다.")
        print(f"{'='*W}\n")