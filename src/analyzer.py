"""
analyzer.py
앙상블 회귀 모델 + 보정 레이어 기반 운동성 % 예측 모듈
v2: IsotonicRegression 보정 레이어 추가
"""

import pickle
import numpy as np


class MotilityAnalyzer:
    """앙상블 회귀 모델 + 보정 레이어 운동성 분석기"""

    FEATURE_COLS = [
        'speed_mean', 'speed_median', 'speed_75', 'speed_90',
        'lin_mean', 'lin_75', 'straight_mean',
        'ratio_fast', 'ratio_medium', 'ratio_slow', 'n_tracks'
    ]

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: motility_ensemble_v2.pkl 경로
                        없으면 자동으로 v2 → v1 순서로 탐색
        """
        import os
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        # v2 → v1 순서로 자동 탐색
        if model_path is None:
            for fname in ['motility_ensemble_v2.pkl',
                          'motility_ensemble.pkl']:
                candidate = os.path.join(base, fname)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.ridge       = data['ridge']
        self.rf          = data['rf']
        self.scaler      = data['scaler']
        self.calibrators = data.get('calibrators', None)
        self.version     = data.get('version', 'v1')

        print(f"✅ MotilityAnalyzer {self.version} 로드 완료: "
              f"{model_path}")

        if self.calibrators:
            print(f"   보정 레이어: 활성화 ✅")
        else:
            print(f"   보정 레이어: 없음 (v1)")

    def predict(self, features: dict) -> dict:
        """
        특징 딕셔너리 → 운동성 % 예측

        단계:
        1. 앙상블 (Ridge + RandomForest) 예측
        2. Isotonic 보정 레이어 적용 (v2)
        3. 합이 100%가 되도록 정규화

        Returns:
            {
              'progressive':     float,
              'non_progressive': float,
              'immotile':        float,
            }
        """
        X    = np.array([[features[c] for c in self.FEATURE_COLS]])
        X_sc = self.scaler.transform(X)

        # 앙상블 예측
        pred = (self.ridge.predict(X_sc)[0] +
                self.rf.predict(X_sc)[0]) / 2

        # v2: 보정 레이어 적용
        if self.calibrators:
            labels = ['progressive', 'non_progressive', 'immotile']
            for i, label in enumerate(labels):
                pred[i] = self.calibrators[label].predict([pred[i]])[0]

        # 0~100 클리핑 + 합 100% 정규화
        pred  = np.clip(pred, 0, 100)
        total = pred.sum()
        if total > 0:
            pred = pred / total * 100

        return {
            'progressive':     round(float(pred[0]), 1),
            'non_progressive': round(float(pred[1]), 1),
            'immotile':        round(float(pred[2]), 1),
        }