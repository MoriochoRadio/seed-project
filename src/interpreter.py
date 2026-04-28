"""
interpreter.py
WHO 기준 기반 해석 + 신뢰도 평가 + 종합 판정
"""

import numpy as np

# ── WHO 6판 (2021) 기준 ───────────────────────────────────
WHO_CRITERIA = {
    'progressive_normal':    32,
    'total_motility_normal': 40,
    'immotile_warning':      50,
    'immotile_severe':       70,
}

KINEMATIC_REFERENCE = {
    'VCL': 25.0, 'VSL': 15.0, 'VAP': 20.0,
    'LIN': 0.50, 'STR': 0.80, 'WOB': 0.70,
    'ALH': 2.0,
}


def interpret_motility(prog: float,
                       non_prog: float,
                       immotile: float) -> dict:
    """
    운동성 수치 → WHO 기준 해석
    일반인도 이해할 수 있는 설명형 문장 생성
    """
    interp = []
    issues = 0

    total_motile = prog + non_prog

    # 전진 운동성 해석
    if prog >= WHO_CRITERIA['progressive_normal']:
        interp.append(
            f"✅ 전진 운동성 {prog:.1f}% — "
            f"정상 (기준 {WHO_CRITERIA['progressive_normal']}% 이상)")
    elif prog >= 22:
        issues += 1
        interp.append(
            f"⚠️  전진 운동성 {prog:.1f}% — "
            f"기준({WHO_CRITERIA['progressive_normal']}%)보다 낮음 "
            f"(전진 운동 정자 비율이 다소 부족)")
    else:
        issues += 2
        interp.append(
            f"🔴 전진 운동성 {prog:.1f}% — "
            f"WHO 정상 기준보다 현저히 낮음 "
            f"(앞으로 나아가는 정자 비율이 크게 부족)")

    # 총 운동성 해석
    if total_motile >= WHO_CRITERIA['total_motility_normal']:
        interp.append(
            f"✅ 총 운동성 {total_motile:.1f}% — "
            f"정상 (기준 {WHO_CRITERIA['total_motility_normal']}% 이상)")
    else:
        issues += 1
        interp.append(
            f"⚠️  총 운동성 {total_motile:.1f}% — "
            f"기준({WHO_CRITERIA['total_motility_normal']}%)보다 낮음 "
            f"(움직이는 정자 전체 비율 부족)")

    # 비운동성 해석
    if immotile >= WHO_CRITERIA['immotile_severe']:
        issues += 2
        interp.append(
            f"🔴 비운동성 {immotile:.1f}% — "
            f"움직이지 않는 정자 비율이 매우 높음 "
            f"(기준 {WHO_CRITERIA['immotile_severe']}% 미만)")
    elif immotile >= WHO_CRITERIA['immotile_warning']:
        issues += 1
        interp.append(
            f"⚠️  비운동성 {immotile:.1f}% — "
            f"움직이지 않는 정자 비율이 높음 "
            f"(기준 {WHO_CRITERIA['immotile_warning']}% 미만)")
    else:
        interp.append(
            f"✅ 비운동성 {immotile:.1f}% — "
            f"정상 범위")

    # 판정 레벨
    if issues == 0:
        level  = 'normal'
        status = '정상 범위'
        recommendation = (
            "AI 분석 결과 운동성 지표가 WHO 정상 기준을 충족합니다.\n"
            "   정확한 진단을 위해 정기적인 병원 검사를 권장합니다.")
    elif issues <= 2:
        level  = 'warning'
        status = '경계 범위'
        recommendation = (
            "일부 지표가 WHO 기준에 미치지 못합니다.\n"
            "   생활 습관 개선(금연, 금주, 규칙적 운동)을 권고하며,\n"
            "   병원 전문의 상담을 받아보시기 바랍니다.")
    else:
        level  = 'severe'
        status = '주의 필요'
        recommendation = (
            "여러 지표가 WHO 기준 이하입니다.\n"
            "   이 결과는 보조 분석이며 의학적 진단이 아닙니다.\n"
            "   정확한 진단을 위해 반드시 비뇨의학과 또는\n"
            "   난임클리닉 전문의를 방문하시기 바랍니다.")

    return {
        'status':         status,
        'level':          level,
        'interpretation': interp,
        'recommendation': recommendation,
        'total_motile':   round(total_motile, 1),
    }


def assess_confidence(counts: list,
                      track_history: dict,
                      N: int) -> dict:
    """
    분석 신뢰도 평가
    """
    score  = 100
    issues = []

    # 탐지 안정성
    if len(counts) > 1:
        cv = float(np.std(counts) / (np.mean(counts) + 1e-6))
        if cv > 0.3:
            score -= 20
            issues.append("프레임 간 탐지 약간 불안정")
        elif cv > 0.15:
            score -= 10

    # 정자 수
    if N < 5:
        score -= 40
        issues.append(
            f"탐지된 정자 수 매우 부족 ({N}개) — "
            f"정자 농도가 매우 낮거나 영상 품질 문제")
    elif N < 10:
        score -= 25
        issues.append(
            f"탐지된 정자 수 부족 ({N}개) — "
            f"정자 농도가 낮을 수 있음")
    elif N < 20:
        score -= 10
        issues.append(f"탐지된 정자 수 적음 ({N}개)")

    # 추적 안정성
    if track_history:
        long_tracks = sum(
            1 for pts in track_history.values()
            if len(pts) >= 10)
        track_ratio = long_tracks / len(track_history)
        if track_ratio < 0.3:
            score -= 20
        elif track_ratio < 0.6:
            score -= 10

    score = max(0, score)

    if score >= 80:
        level = '높음'
        icon  = '✅'
        needs_retake = False
    elif score >= 60:
        level = '보통'
        icon  = '⚠️ '
        needs_retake = False
    else:
        level = '낮음 — 결과 신뢰도 부족'
        icon  = '🔴'
        needs_retake = True

    return {
        'confidence_score': score,
        'confidence_level': level,
        'confidence_icon':  icon,
        'quality_issues':   issues,
        'needs_retake':     needs_retake,
    }


def interpret_kinematics(kinematics: dict) -> dict:
    """
    키네마틱 파라미터 → 설명형 해석
    """
    if not kinematics:
        return {}

    interps = []
    issues  = 0

    vcl  = kinematics.get('VCL_mean', 0)
    vsl  = kinematics.get('VSL_mean', 0)
    vap  = kinematics.get('VAP_mean', 0)
    lin  = kinematics.get('LIN_mean', 0)
    str_ = kinematics.get('STR_mean', 0)

    # VCL
    if vcl >= 25:
        interps.append(
            f"✅ VCL {vcl:.1f} µm/s — "
            f"곡선 속도 정상 (기준 ≥25)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VCL {vcl:.1f} µm/s — "
            f"곡선 속도 낮음 (기준 ≥25)")

    # VSL
    if vsl >= 15:
        interps.append(
            f"✅ VSL {vsl:.1f} µm/s — "
            f"직선 속도 정상 (기준 ≥15)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VSL {vsl:.1f} µm/s — "
            f"직선 속도 낮음 (기준 ≥15)")

    # VAP
    if vap >= 20:
        interps.append(
            f"✅ VAP {vap:.1f} µm/s — "
            f"평균 경로 속도 정상 (기준 ≥20)")
    else:
        issues += 1
        interps.append(
            f"⚠️  VAP {vap:.1f} µm/s — "
            f"평균 경로 속도 낮음 (기준 ≥20)")

    # STR
    if str_ >= 0.80:
        interps.append(
            f"✅ STR {str_:.3f} — "
            f"직진성 정상 (기준 ≥0.80)")
    elif str_ >= 0.60:
        issues += 1
        interps.append(
            f"⚠️  STR {str_:.3f} — "
            f"직진성 경계 (기준 ≥0.80)")
    else:
        issues += 2
        interps.append(
            f"🔴 STR {str_:.3f} — "
            f"직진성 낮음 (기준 ≥0.80)")

    # LIN
    if lin >= 0.50:
        interps.append(
            f"✅ LIN {lin:.3f} — "
            f"직선성 정상 (기준 ≥0.50)")
    else:
        issues += 1
        interps.append(
            f"⚠️  LIN {lin:.3f} — "
            f"직선성 낮음 (기준 ≥0.50)")

    ppms = (vap >= 25) and (str_ >= 0.80)
    ppms_note = (
        "전진 운동성 정자(PPMS) 기준 충족 가능"
        if ppms else
        "전진 운동성 정자(PPMS) 기준 미달 가능성")

    return {
        'interpretations': interps,
        'ppms_note':        ppms_note,
        'kinematic_level':  'normal' if issues == 0
                            else 'warning' if issues <= 2
                            else 'poor',
        'issues':           issues,
    }


def get_overall_judgment(motility_level: str,
                         morph_level: str,
                         confidence_score: int) -> dict:
    """
    운동성 + 형태 + 신뢰도 → 종합 판정
    일반인이 이해하기 쉬운 종합 메시지 생성
    """
    # 신뢰도 낮으면 결과 신뢰도 경고
    if confidence_score < 60:
        return {
            'overall_level':   'uncertain',
            'overall_status':  '재검사 필요',
            'overall_icon':    '🔄',
            'overall_message': (
                "영상 품질 또는 정자 수 부족으로 신뢰도 있는\n"
                "   분석이 어렵습니다. 재촬영을 권고합니다."),
        }

    # 레벨 점수화
    level_score = {
        'normal': 0, 'warning': 1,
        'severe': 2, 'poor': 2, '': 0
    }

    mot_score  = level_score.get(motility_level, 0)
    morph_score = level_score.get(morph_level, 0)
    total       = mot_score + morph_score

    if total == 0:
        return {
            'overall_level':   'normal',
            'overall_status':  '전반적 정상',
            'overall_icon':    '🟢',
            'overall_message': (
                "운동성과 형태 모두 WHO 기준 정상 범위입니다.\n"
                "   현재 상태를 유지하시고 정기 검진을 권장합니다."),
        }
    elif total <= 2:
        return {
            'overall_level':   'warning',
            'overall_status':  '일부 주의',
            'overall_icon':    '🟡',
            'overall_message': (
                "일부 지표가 정상 기준에 미치지 못합니다.\n"
                "   생활 습관 개선과 전문의 상담을 권고합니다.\n"
                "   임신을 계획 중이라면 난임클리닉 방문을 고려하세요."),
        }
    else:
        return {
            'overall_level':   'severe',
            'overall_status':  '전문의 상담 권고',
            'overall_icon':    '🔴',
            'overall_message': (
                "여러 지표에서 주의가 필요한 결과가 나왔습니다.\n"
                "   이 결과는 AI 보조 분석이며 의학적 진단이 아닙니다.\n"
                "   정확한 진단을 위해 비뇨의학과 또는\n"
                "   난임클리닉 전문의 방문을 강력히 권고합니다."),
        }