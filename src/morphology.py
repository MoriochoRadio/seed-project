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

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """YOLO 크롭 이미지 → (3,128,128) 정규화 텐서 (배치 stack용)."""
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        resized = cv2.resize(gray, (128, 128))
        img     = np.stack([resized, resized, resized], axis=0)
        img     = torch.tensor(img / 255.0, dtype=torch.float32)
        return self.transform(img)

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """YOLO 크롭 이미지 → 모델 입력 텐서 (1,3,128,128)"""
        return self._to_tensor(crop).unsqueeze(0)

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

    def classify_batch(self, crops: list,
                       batch_size: int = 128) -> np.ndarray:
        """크롭 리스트 → 부위별 시그모이드 확률 (N,4) 배열.

        크롭을 한 개씩 추론하던 것을 배치로 묶어 GPU 활용도를 높인다.
        결과는 한 개씩 추론과 동일하며 속도만 빨라진다.
        """
        if not crops:
            return np.zeros((0, 4), dtype=np.float32)

        all_probs = []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            batch = torch.stack(
                [self._to_tensor(c) for c in chunk]).to(self.device)
            with torch.no_grad():
                out   = self.model(batch)
                probs = torch.sigmoid(out).cpu().numpy()  # (B,4)
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def analyze_crops(self, crops: list,
                      batch_size: int = 128) -> dict:
        """여러 정자 크롭 → 샘플 수준 형태 분석 (배치 추론)."""
        if not crops:
            return {}

        parts = ['head', 'acrosome', 'vacuole', 'tail']
        probs = self.classify_batch(crops, batch_size)      # (N,4)
        thr   = np.array([self.THRESHOLDS[p] for p in parts])
        preds = (probs > thr).astype(int)                   # (N,4)

        n_total   = int(preds.shape[0])
        n_normal  = int((preds.sum(axis=1) == 0).sum())
        abnormal_rates = {
            p: round(float(preds[:, i].sum()) / n_total * 100, 1)
            for i, p in enumerate(parts)
        }

        return {
            'n_analyzed':     n_total,
            'n_normal':       n_normal,
            'normal_rate':    round(n_normal / n_total * 100, 1),
            'abnormal_rates': abnormal_rates,
        }

    def analyze_per_tid(self, crops_per_tid: dict,
                        batch_size: int = 128) -> dict:
        """{tid: [crops]} → {tid: 형태 요약} (track별 배치 추론).

        각 track(정자 1마리)의 크롭들을 모아 부위별 이상 여부를
        다수결로 집계하고, 모든 부위가 정상일 때 is_normal=True.
        """
        if not crops_per_tid:
            return {}

        parts  = ['head', 'acrosome', 'vacuole', 'tail']
        thr    = np.array([self.THRESHOLDS[p] for p in parts])
        result = {}

        for tid, crops in crops_per_tid.items():
            if not crops:
                continue
            probs = self.classify_batch(crops, batch_size)   # (n,4)
            preds = (probs > thr).astype(int)                # (n,4)
            n     = int(preds.shape[0])

            # 부위별: track 내 크롭의 과반이 이상이면 이상으로 판정
            part_abnormal = {
                p: bool(preds[:, i].sum() * 2 > n)
                for i, p in enumerate(parts)
            }
            is_normal = not any(part_abnormal.values())

            result[tid] = {
                'n_crops':       n,
                'is_normal':     is_normal,
                'part_abnormal': part_abnormal,
                'mean_probs': {
                    p: round(float(probs[:, i].mean()), 3)
                    for i, p in enumerate(parts)
                },
            }
        return result

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
