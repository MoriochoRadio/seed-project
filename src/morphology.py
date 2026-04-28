"""
morphology.py
EfficientNet-B3 기반 정자 형태 분석 모듈 (v3)

학습 데이터: MHSMA 전체 1,540개
개선 사항:
  - VISEM 도메인 증강 (모션블러, 노이즈, 저해상도 시뮬레이션)
  - Focal Loss + WeightedRandomSampler
  - Early Stopping (Epoch 31)

성능:
  MHSMA 테스트셋 AUC: 0.725 (head 0.677 / acrosome 0.711 /
                               vacuole 0.767 / tail 0.752)
  VISEM 실제 적용:    정상 형태율 ~8-12% (v2 대비 현실적)

주의:
  MHSMA(학습) ↔ VISEM(적용) 도메인 차이 존재
  수치는 참고용이며 추세 판단에 활용 권장
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class SpermMorphologyModel(nn.Module):
    """EfficientNet-B3 멀티태스크 형태 분류 모델"""

    def __init__(self):
        super().__init__()
        backbone      = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.shared   = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head_cls     = nn.Linear(256, 1)
        self.acrosome_cls = nn.Linear(256, 1)
        self.vacuole_cls  = nn.Linear(256, 1)
        self.tail_cls     = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        return torch.cat([
            self.head_cls(x),
            self.acrosome_cls(x),
            self.vacuole_cls(x),
            self.tail_cls(x),
        ], dim=1)


class MorphologyAnalyzer:
    """정자 형태 분석기 (v3 — VISEM 도메인 적응)"""

    # 검증셋 기반 최적 임계값 (v3)
    THRESHOLDS = {
        'head':     0.58,
        'acrosome': 0.52,
        'vacuole':  0.62,
        'tail':     0.58,
    }

    PART_KR = {
        'head':     '머리',
        'acrosome': '첨체',
        'vacuole':  '공포',
        'tail':     '꼬리',
    }

    WHO_NORMAL_MORPHOLOGY = 4.0

    def __init__(self, model_path: str):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpermMorphologyModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path,
                       map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
        ])
        print(f"✅ MorphologyAnalyzer v3 로드 완료: {model_path}")
        print(f"   MHSMA AUC: 0.725 / "
              f"VISEM 정상형태율: ~8-12% (현실적)")

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """YOLO 크롭 이미지 → 모델 입력 텐서"""
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        resized = cv2.resize(gray, (128, 128))
        img     = np.stack([resized, resized, resized], axis=0)
        img     = torch.tensor(img / 255.0, dtype=torch.float32)
        return self.transform(img).unsqueeze(0)

    def classify_single(self, crop: np.ndarray) -> dict:
        """정자 크롭 하나 → 4가지 부위 정상/비정상 분류"""
        tensor = self.preprocess_crop(crop).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.sigmoid(output).cpu().numpy()[0]

        parts = ['head', 'acrosome', 'vacuole', 'tail']
        preds = {
            p: int(probs[i] > self.THRESHOLDS[p])
            for i, p in enumerate(parts)
        }
        preds['is_normal'] = all(v == 0 for v in preds.values())
        preds['probs'] = {
            p: round(float(probs[i]), 3)
            for i, p in enumerate(parts)
        }
        return preds

    def analyze_crops(self, crops: list) -> dict:
        """여러 정자 크롭 → 샘플 수준 형태 분석"""
        if not crops:
            return {}

        results  = [self.classify_single(c) for c in crops]
        n_total  = len(results)
        n_normal = sum(1 for r in results if r['is_normal'])

        parts = ['head', 'acrosome', 'vacuole', 'tail']
        abnormal_rates = {
            p: round(
                sum(1 for r in results if r[p] == 1)
                / n_total * 100, 1)
            for p in parts
        }

        return {
            'n_analyzed':     n_total,
            'n_normal':       n_normal,
            'normal_rate':    round(n_normal / n_total * 100, 1),
            'abnormal_rates': abnormal_rates,
        }

    def interpret_morphology(self, morphology: dict) -> dict:
        """형태 분석 결과 → WHO 기준 해석"""
        if not morphology:
            return {}

        normal_rate = morphology['normal_rate']

        if normal_rate >= 14:
            level, status, icon = 'normal',  '정상 범위', '✅'
        elif normal_rate >= 4:
            level, status, icon = 'warning', '경계 범위', '⚠️ '
        else:
            level, status, icon = 'poor',    '주의 필요', '🔴'

        interps = [
            f"{icon} 정상 형태율 {normal_rate:.1f}% "
            f"(WHO 기준 ≥4%)  ※참고용"
        ]

        rates = morphology['abnormal_rates']
        thresholds = {
            'head': 40, 'acrosome': 40,
            'vacuole': 30, 'tail': 20
        }
        for p, thr in thresholds.items():
            if rates[p] >= thr:
                kr = self.PART_KR[p]
                interps.append(
                    f"⚠️  {kr} 이상 {rates[p]:.1f}% — 높음")

        return {
            'level':           level,
            'status':          status,
            'interpretations': interps,
        }
