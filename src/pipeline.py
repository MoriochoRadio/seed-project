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
                           interpret_kinematics)


class SpermAnalysisPipeline:
    """AI-CASA 통합 파이프라인 (운동성 + 키네마틱 + 형태)"""

    def __init__(self,
                 yolo_path    : str = None,
                 model_path   : str = None,
                 tracker_config: str = None,
                 morph_path   : str = None):

        base = r'C:\Users\neo62\sperm-ai'

        yolo_path      = yolo_path or os.path.join(
            base, 'models', 'yolo11_sperm_v2', 'weights', 'best.pt')
        model_path     = model_path or os.path.join(
            base, 'models', 'motility_ensemble.pkl')
        tracker_config = tracker_config or os.path.join(
            base, 'bytetrack_custom.yaml')
        morph_path     = morph_path or os.path.join(
            base, 'models', 'morphology_efficientnet_b3_v3.pt')

        self.detector  = SpermDetector(yolo_path)
        self.tracker   = SpermTracker(self.detector, tracker_config)
        self.analyzer  = MotilityAnalyzer(model_path)

        # 형태 분석기 (없으면 운동성만 수행)
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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
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

        Returns:
            전체 CASA 분석 결과 딕셔너리
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
        morphology       = {}
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

        return {
            'video':             name,
            'N':                 N,
            **motility,
            **interp,
            **quality,
            'kinematics':        kinematics,
            'morphology':        morphology,
            'morphology_interp': morphology_interp,
        }

    # ── 보고서 출력 ────────────────────────────────────────
    @staticmethod
    def print_report(result: dict,
                     participant_id: str = None) -> None:
        """AI-CASA 통합 보고서 출력"""

        title = "AI-CASA 분석 보고서"
        if participant_id:
            title += f" — 참가자 {participant_id}"

        level_emoji = {
            'normal': '🟢', 'warning': '🟡', 'severe': '🔴'}
        emoji = level_emoji.get(result['level'], '⚪')

        print(f"\n{'='*57}")
        print(f"  {title}")
        print(f"{'='*57}")

        # 신뢰도
        print(f"\n📊 분석 신뢰도: "
              f"{result['confidence_icon']} "
              f"{result['confidence_level']} "
              f"({result['confidence_score']}점)")
        for issue in result['quality_issues']:
            print(f"   ⚠️  {issue}")
        if result['needs_retake']:
            print(f"   🔄 재촬영을 권고합니다")

        # 운동성
        print(f"\n📈 운동성 분석 (탐지 정자 수: {result['N']}개)")
        print(f"   전진 운동성:   {result['progressive']:>5.1f}%")
        print(f"   비전진 운동성: {result['non_progressive']:>5.1f}%")
        print(f"   비운동성:      {result['immotile']:>5.1f}%")
        print(f"   총 운동성:     {result['total_motile']:>5.1f}%")

        # WHO 해석
        print(f"\n🔍 WHO 기준 해석")
        for interp in result['interpretation']:
            print(f"   {interp}")

        print(f"\n{emoji} 운동성 판정: [{result['status']}]")
        print(f"\n💬 권고사항:")
        print(f"   {result['recommendation']}")

        # 키네마틱
        if result.get('kinematics'):
            k    = result['kinematics']
            from .interpreter import interpret_kinematics
            kint = interpret_kinematics(k)

            print(f"\n📐 CASA 키네마틱 파라미터")
            print(f"   {'파라미터':<8} {'측정값':>10}  참고 기준")
            print(f"   {'-'*38}")
            print(f"   {'VCL':<8} {k['VCL_mean']:>7.1f} µm/s  ≥ 25")
            print(f"   {'VSL':<8} {k['VSL_mean']:>7.1f} µm/s  ≥ 15")
            print(f"   {'VAP':<8} {k['VAP_mean']:>7.1f} µm/s  ≥ 20")
            print(f"   {'LIN':<8} {k['LIN_mean']:>7.3f}       ≥ 0.50")
            print(f"   {'STR':<8} {k['STR_mean']:>7.3f}       ≥ 0.80")
            print(f"   {'WOB':<8} {k['WOB_mean']:>7.3f}       ≥ 0.70")
            print(f"   {'ALH':<8} {k['ALH_mean']:>7.2f} µm    ≥ 2.0")
            print(f"   (분석 정자: {k['n_analyzed']}개)")

            if kint:
                print(f"\n🔬 키네마틱 해석")
                for line in kint['interpretations']:
                    print(f"   {line}")
                print(f"   📌 {kint['ppms_note']}")

        # 형태 분석
        if result.get('morphology') and result['morphology']:
            m    = result['morphology']
            mint = result.get('morphology_interp', {})

            print(f"\n🧬 형태 분석 결과")
            print(f"   ※ 참고용 (도메인 차이로 수치 과대 추정 가능)")
            print(f"   {'부위':<12} {'비정상률':>8}  상태")
            print(f"   {'-'*32}")
            for part, kr in [('head','머리'), ('acrosome','첨체'),
                              ('vacuole','공포'), ('tail','꼬리')]:
                rate = m['abnormal_rates'][part]
                flag = '⚠️ ' if rate >= 50 else '✅'
                print(f"   {kr}({part:<10}) {rate:>6.1f}%  {flag}")

            status = mint.get('status', '-')
            print(f"\n   정상 형태율: {m['normal_rate']:.1f}%"
                  f"  →  {status}")
            for line in mint.get('interpretations', []):
                print(f"   {line}")

        print(f"\n{'─'*57}")
        print(f"   ※ AI 보조 분석 / 의학적 진단 아님")
        print(f"{'='*57}\n")