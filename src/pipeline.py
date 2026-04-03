"""
pipeline.py
정자 운동성 분석 통합 파이프라인
사용법:
    from src.pipeline import SpermAnalysisPipeline
    pipeline = SpermAnalysisPipeline()
    result   = pipeline.analyze('video.mp4')
    pipeline.print_report(result)
"""

from .detector    import SpermDetector
from .tracker     import SpermTracker
from .analyzer    import MotilityAnalyzer
from .interpreter import interpret_motility, assess_confidence


class SpermAnalysisPipeline:
    """정자 운동성 분석 통합 파이프라인"""

    def __init__(self,
                 yolo_path    : str = None,
                 model_path   : str = None,
                 tracker_config: str = None):

        import os
        base = r'C:\Users\neo62\sperm-ai'

        yolo_path     = yolo_path or os.path.join(
            base, 'models', 'yolo11_sperm_v2',
            'weights', 'best.pt')
        model_path = model_path or os.path.join(
            base, 'models', 'motility_ensemble.pkl')
        tracker_config = tracker_config or os.path.join(
            base, 'bytetrack_custom.yaml')

        self.detector  = SpermDetector(yolo_path)
        self.tracker   = SpermTracker(self.detector, tracker_config)
        self.analyzer  = MotilityAnalyzer(model_path)

    def analyze(self, video_path: str,
                verbose: bool = True) -> dict:
        """
        영상 하나를 입력받아 전체 분석 수행

        Args:
            video_path: 분석할 영상 경로
            verbose: 진행 상황 출력 여부

        Returns:
            전체 분석 결과 딕셔너리
        """
        name = video_path.split('\\')[-1]
        if verbose:
            print(f"분석 시작: {name}")
            print("─" * 50)

        # 1. 전체 정자 수 추정
        N, counts = self.detector.estimate_sperm_count(video_path)
        if verbose:
            print(f"[1/4] 정자 수 기준값: {N}개")

        # 2. 추적
        track_history = self.tracker.track_video(video_path)
        if verbose:
            print(f"[2/4] 추적 완료: {len(track_history)}개 ID")

        # 3. 특징 추출
        features = self.tracker.extract_features(track_history)
        if features is None:
            return None
        if verbose:
            print(f"[3/4] 특징 추출 완료")

        # 4. 운동성 예측
        motility = self.analyzer.predict(features)
        if verbose:
            print(f"[4/4] 분석 완료")

        # 5. WHO 해석
        interp  = interpret_motility(
            motility['progressive'],
            motility['non_progressive'],
            motility['immotile'])

        # 6. 신뢰도 평가
        quality = assess_confidence(counts, track_history, N)

        return {
            'video':           name,
            'N':               N,
            **motility,
            **interp,
            **quality,
        }

    @staticmethod
    def print_report(result: dict,
                     participant_id: str = None) -> None:
        """분석 결과 보고서 출력"""

        title = "정자 운동성 분석 보고서"
        if participant_id:
            title += f" — 참가자 {participant_id}"

        level_emoji = {
            'normal': '🟢', 'warning': '🟡', 'severe': '🔴'}
        emoji = level_emoji.get(result['level'], '⚪')

        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")

        print(f"\n📊 분석 신뢰도: "
              f"{result['confidence_icon']} "
              f"{result['confidence_level']} "
              f"({result['confidence_score']}점)")

        for issue in result['quality_issues']:
            print(f"   ⚠️  {issue}")

        if result['needs_retake']:
            print(f"   🔄 재촬영을 권고합니다")

        print(f"\n📈 운동성 분석 결과 (탐지 정자 수: {result['N']}개)")
        print(f"   전진 운동성:   {result['progressive']:>5.1f}%")
        print(f"   비전진 운동성: {result['non_progressive']:>5.1f}%")
        print(f"   비운동성:      {result['immotile']:>5.1f}%")
        print(f"   총 운동성:     {result['total_motile']:>5.1f}%")

        print(f"\n🔍 WHO 기준 해석")
        for interp in result['interpretation']:
            print(f"   {interp}")

        print(f"\n{emoji} 종합 판정: [{result['status']}]")
        print(f"\n💬 권고사항:")
        print(f"   {result['recommendation']}")
        print(f"\n{'─'*55}")
        print(f"   ※ 이 결과는 AI 보조 분석이며 의학적 진단이 아닙니다.")
        print(f"{'='*55}\n")