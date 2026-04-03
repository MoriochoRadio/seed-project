"""
interpreter.py
WHO 기준 해석 + 신뢰도 평가 모듈
"""

import numpy as np


# WHO 6판 (2021) 기준
WHO_CRITERIA = {
    'progressive_normal':    32,
    'total_motility_normal': 40,
    'immotile_warning':      50,
    'immotile_severe':       70,
}


def interpret_motility(prog: float,
                       non_prog: float,
                       immotile: float) -> dict:
    """
    WHO 기준으로 운동성 수치 해석

    Returns:
        status, level, interpretation, recommendation
    """
    total_motile = prog + non_prog
    issues       = []
    interpretation = []

    # 전진 운동성
    if prog >= WHO_CRITERIA['progressive_normal']:
        interpretation.append(
            f"✅ 전진 운동성 {prog:.1f}% — "
            f"WHO 정상 기준({WHO_CRITERIA['progressive_normal']}%) 충족")
    elif prog >= WHO_CRITERIA['progressive_normal'] * 0.7:
        issues.append('전진 운동성 경계')
        interpretation.append(
            f"⚠️  전진 운동성 {prog:.1f}% — "
            f"WHO 정상 기준({WHO_CRITERIA['progressive_normal']}%) 미달")
    else:
        issues.append('전진 운동성 저하')
        interpretation.append(
            f"🔴 전진 운동성 {prog:.1f}% — "
            f"WHO 정상 기준보다 현저히 낮음")

    # 총 운동성
    if total_motile >= WHO_CRITERIA['total_motility_normal']:
        interpretation.append(
            f"✅ 총 운동성 {total_motile:.1f}% — "
            f"WHO 정상 기준({WHO_CRITERIA['total_motility_normal']}%) 충족")
    else:
        issues.append('총 운동성 저하')
        interpretation.append(
            f"⚠️  총 운동성 {total_motile:.1f}% — "
            f"WHO 정상 기준({WHO_CRITERIA['total_motility_normal']}%) 미달")

    # 비운동성
    if immotile >= WHO_CRITERIA['immotile_severe']:
        issues.append('비운동성 심각')
        interpretation.append(
            f"🔴 비운동성 {immotile:.1f}% — "
            f"매우 높음 (기준 {WHO_CRITERIA['immotile_severe']}% 초과)")
    elif immotile >= WHO_CRITERIA['immotile_warning']:
        issues.append('비운동성 주의')
        interpretation.append(
            f"⚠️  비운동성 {immotile:.1f}% — "
            f"주의 필요 (기준 {WHO_CRITERIA['immotile_warning']}% 초과)")
    else:
        interpretation.append(
            f"✅ 비운동성 {immotile:.1f}% — 정상 범위")

    # 종합 판정
    if not issues:
        level  = 'normal'
        status = '정상 범위'
        recommendation = (
            "AI 분석 결과 WHO 기준 정상 범위입니다. "
            "정확한 진단을 위해 정기적인 병원 검사를 권장합니다.")
    elif '비운동성 심각' in issues or '전진 운동성 저하' in issues:
        level  = 'severe'
        status = '주의 필요'
        recommendation = (
            "AI 분석 결과 일부 지표가 WHO 기준 이하입니다. "
            "정밀 검사를 위해 병원 방문을 권고합니다. "
            "이 결과는 보조 분석이며 의학적 진단을 대체하지 않습니다.")
    else:
        level  = 'warning'
        status = '경계 범위'
        recommendation = (
            "AI 분석 결과 일부 지표가 경계 범위입니다. "
            "추가 확인을 위해 병원 검사를 고려해보세요. "
            "이 결과는 보조 분석이며 의학적 진단을 대체하지 않습니다.")

    return {
        'status':         status,
        'level':          level,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'total_motile':   round(total_motile, 1),
    }


def assess_confidence(counts: list,
                      track_history: dict,
                      N: int) -> dict:
    """
    영상 품질 + 분석 신뢰도 평가

    Returns:
        confidence_score, confidence_level, quality_issues,
        needs_retake
    """
    quality_issues   = []
    confidence_score = 100

    # 1. 탐지 안정성
    if len(counts) > 1:
        cv = np.std(counts) / (np.mean(counts) + 1e-6)
        if cv > 0.3:
            quality_issues.append(
                "프레임 간 탐지 불안정 (영상 흔들림 가능성)")
            confidence_score -= 20
        elif cv > 0.15:
            quality_issues.append("프레임 간 탐지 약간 불안정")
            confidence_score -= 10

    # 2. 정자 수
    if N < 5:
        quality_issues.append(
            f"탐지된 정자 수 매우 부족 ({N}개) — "
            f"정자 농도가 매우 낮거나 영상 품질 문제")
        confidence_score -= 40
    elif N < 10:
        quality_issues.append(
            f"탐지된 정자 수 부족 ({N}개) — "
            f"정자 농도가 낮을 수 있음")
        confidence_score -= 25
    elif N < 20:
        quality_issues.append(f"탐지된 정자 수 적음 ({N}개)")
        confidence_score -= 10

    # 3. 추적 안정성
    n_tracks    = len([tid for tid, pts in track_history.items()
                       if len(pts) >= 5])
    track_ratio = n_tracks / (N + 1e-6)

    if track_ratio < 0.3:
        quality_issues.append("추적 안정성 낮음")
        confidence_score -= 20
    elif track_ratio < 0.6:
        quality_issues.append("추적 안정성 보통")
        confidence_score -= 10

    confidence_score = max(0, min(100, confidence_score))

    if confidence_score >= 80:
        level = "높음"
        icon  = "✅"
    elif confidence_score >= 60:
        level = "보통"
        icon  = "⚠️"
    else:
        level = "낮음 — 결과 신뢰도 부족"
        icon  = "🔴"

    return {
        'confidence_score': confidence_score,
        'confidence_level': level,
        'confidence_icon':  icon,
        'quality_issues':   quality_issues,
        'needs_retake':     confidence_score < 60,
    }

# CASA 키네마틱 참고 기준
# (WHO 5판 + 임상 연구 기반, 장비별 차이 있음)
KINEMATIC_REFERENCE = {
    'VCL': {'normal': 25.0,  'unit': 'µm/s', 'name': '곡선 속도'},
    'VSL': {'normal': 15.0,  'unit': 'µm/s', 'name': '직선 속도'},
    'VAP': {'normal': 20.0,  'unit': 'µm/s', 'name': '평균 경로 속도'},
    'LIN': {'normal': 0.50,  'unit': '',      'name': '직선성'},
    'STR': {'normal': 0.80,  'unit': '',      'name': '직진성'},
    'WOB': {'normal': 0.70,  'unit': '',      'name': '진동도'},
    'ALH': {'normal': 2.0,   'unit': 'µm',    'name': '횡변위 진폭'},
}


def interpret_kinematics(kinematics: dict) -> dict:
    """
    CASA 키네마틱 파라미터 해석

    Args:
        kinematics: compute_sample_kinematics() 결과

    Returns:
        {
          'interpretations': [해석 문자열 리스트],
          'ppms': float,         전진 운동성 정자 비율 추정
          'kinematic_level': str normal / warning / poor
        }
    """
    if not kinematics:
        return {}

    interps = []
    issues  = 0

    vcl = kinematics.get('VCL_mean', 0)
    vsl = kinematics.get('VSL_mean', 0)
    vap = kinematics.get('VAP_mean', 0)
    lin = kinematics.get('LIN_mean', 0)
    str_ = kinematics.get('STR_mean', 0)
    wob = kinematics.get('WOB_mean', 0)
    alh = kinematics.get('ALH_mean', 0)

    # VCL 해석
    if vcl >= 25:
        interps.append(
            f"✅ VCL {vcl:.1f} µm/s — 곡선 속도 정상 (기준 ≥25)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VCL {vcl:.1f} µm/s — 곡선 속도 낮음 (기준 ≥25)")

    # VSL 해석
    if vsl >= 15:
        interps.append(
            f"✅ VSL {vsl:.1f} µm/s — 직선 속도 정상 (기준 ≥15)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VSL {vsl:.1f} µm/s — 직선 속도 낮음 (기준 ≥15)")

    # VAP 해석
    if vap >= 20:
        interps.append(
            f"✅ VAP {vap:.1f} µm/s — 평균 경로 속도 정상 (기준 ≥20)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VAP {vap:.1f} µm/s — 평균 경로 속도 낮음 (기준 ≥20)")

    # STR 해석 (가장 임상적으로 중요)
    if str_ >= 0.80:
        interps.append(
            f"✅ STR {str_:.3f} — 직진성 정상 (기준 ≥0.80)")
    elif str_ >= 0.60:
        issues += 1
        interps.append(
            f"⚠️  STR {str_:.3f} — 직진성 경계 (기준 ≥0.80)")
    else:
        issues += 2
        interps.append(
            f"🔴 STR {str_:.3f} — 직진성 낮음 (기준 ≥0.80)")

    # LIN 해석
    if lin >= 0.50:
        interps.append(
            f"✅ LIN {lin:.3f} — 직선성 정상 (기준 ≥0.50)")
    else:
        issues += 1
        interps.append(
            f"⚠️  LIN {lin:.3f} — 직선성 낮음 (기준 ≥0.50)")

    # PPMS 추정 (VAP > 25 AND STR > 0.8)
    ppms_eligible = (vap >= 25) and (str_ >= 0.80)
    ppms_note = (
        "전진 운동성 정자(PPMS) 기준 충족 가능"
        if ppms_eligible
        else "전진 운동성 정자(PPMS) 기준 미달 가능성"
    )

    # 종합 등급
    if issues == 0:
        level = 'normal'
    elif issues <= 2:
        level = 'warning'
    else:
        level = 'poor'

    return {
        'interpretations': interps,
        'ppms_note':       ppms_note,
        'kinematic_level': level,
        'issues':          issues,
    }