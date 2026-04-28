/* ═══════════════════════════════════════════════════
   AI-CASA Result Visualization
   ─ 운동성 도넛 차트 (Canvas API)
═══════════════════════════════════════════════════ */

const canvas = document.getElementById('motility-donut');
if (canvas) {
    const ctx  = canvas.getContext('2d');
    const W    = canvas.width;
    const H    = canvas.height;
    const cx   = W / 2;
    const cy   = H / 2;
    const r    = 90;
    const lineWidth = 30;

    const data = window.MOTILITY_DATA;
    const total = data.progressive + data.non_progressive + data.immotile;

    // 백분율 계산 (합 100% 보정)
    const segments = [
        {
            label: 'progressive',
            value: data.progressive / total,
            color: '#2DD4BF'   // 청록 (전진)
        },
        {
            label: 'non_progressive',
            value: data.non_progressive / total,
            color: '#60A5FA'   // 파랑 (비전진)
        },
        {
            label: 'immotile',
            value: data.immotile / total,
            color: '#475569'   // 회색 (비운동)
        },
    ];

    // 배경 트랙
    ctx.beginPath();
    ctx.strokeStyle = '#1A2638';
    ctx.lineWidth   = lineWidth;
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // 세그먼트 그리기 (애니메이션)
    let startAngle = -Math.PI / 2;
    let currentSegment = 0;
    let progress = 0;

    function animate() {
        if (currentSegment >= segments.length) return;

        const seg = segments[currentSegment];
        const segLength = seg.value * Math.PI * 2;
        const segProgress = Math.min(progress, segLength);

        // 전체 다시 그리기
        ctx.clearRect(0, 0, W, H);

        // 배경
        ctx.beginPath();
        ctx.strokeStyle = '#1A2638';
        ctx.lineWidth   = lineWidth;
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        // 이전 완성 세그먼트들
        let curStart = -Math.PI / 2;
        for (let i = 0; i < currentSegment; i++) {
            const len = segments[i].value * Math.PI * 2;
            ctx.beginPath();
            ctx.strokeStyle = segments[i].color;
            ctx.lineCap     = 'butt';
            ctx.lineWidth   = lineWidth;
            ctx.arc(cx, cy, r, curStart, curStart + len);
            ctx.stroke();
            curStart += len;
        }

        // 현재 진행 중 세그먼트
        ctx.beginPath();
        ctx.strokeStyle = seg.color;
        ctx.lineWidth   = lineWidth;
        ctx.arc(cx, cy, r, curStart, curStart + segProgress);
        ctx.stroke();

        progress += 0.04;
        if (progress >= segLength) {
            currentSegment++;
            progress = 0;
        }
        requestAnimationFrame(animate);
    }

    setTimeout(animate, 200);
}
/* ═══════════════════════════════════════════════════
   모드 토글 (일반인용 / 전문가용)
═══════════════════════════════════════════════════ */

const modeButtons = document.querySelectorAll('.mode-btn');
const modeHint    = document.getElementById('mode-hint');
const body        = document.body;

// 페이지 로드 시 저장된 모드 복원 (기본: simple)
const savedMode = localStorage.getItem('aicasa-mode') || 'simple';
applyMode(savedMode);

modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        applyMode(mode);
        localStorage.setItem('aicasa-mode', mode);
    });
});

function applyMode(mode) {
    // 버튼 활성화 표시
    modeButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // body에 클래스 추가 → CSS로 표시 제어
    body.classList.remove('mode-simple', 'mode-expert');
    body.classList.add('mode-' + mode);

    // 힌트 메시지 변경
    if (mode === 'simple') {
        modeHint.innerHTML =
            '💡 일반인용 모드입니다. 각 수치의 의미를 자세히 설명합니다.';
    } else {
        modeHint.innerHTML =
            '🔬 전문가용 모드입니다. 핵심 수치와 WHO 기준 비교 위주로 표시됩니다.';
    }
}
/* ═══════════════════════════════════════════════════
   AI 신뢰도 박스 — 펼치기/접기
═══════════════════════════════════════════════════ */

const trustToggle = document.getElementById('ai-trust-toggle');
const trustDetail = document.getElementById('ai-trust-detail');

if (trustToggle && trustDetail) {
    // 기본은 접힌 상태
    trustDetail.classList.add('collapsed');
    trustToggle.classList.add('collapsed');

    trustToggle.addEventListener('click', () => {
        const isCollapsed = trustDetail.classList.contains('collapsed');
        if (isCollapsed) {
            trustDetail.classList.remove('collapsed');
            trustToggle.classList.remove('collapsed');
        } else {
            trustDetail.classList.add('collapsed');
            trustToggle.classList.add('collapsed');
        }
    });
}