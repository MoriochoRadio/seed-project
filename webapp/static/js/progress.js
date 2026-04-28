/* ═══════════════════════════════════════════════════
   AI-CASA Progress Polling
   ─ 1.5초마다 분석 상태를 확인
   ─ 진행률, 단계, 결과 페이지 자동 이동
═══════════════════════════════════════════════════ */

const stageText    = document.getElementById('stage-text');
const percentText  = document.getElementById('percent-text');
const progressFill = document.getElementById('progress-fill');
const errorMsg     = document.getElementById('error-message');
const errorDetail  = document.getElementById('error-detail');

const stageItems = document.querySelectorAll('.stage-item');

let pollInterval = null;

// ── 시작 시 오류 박스 확실히 숨김 ────────────────────────
errorMsg.hidden = true;
errorMsg.style.display = 'none';

// ── 상태 폴링 ────────────────────────────────────────────
async function pollStatus() {
    try {
        const response = await fetch(`/api/status/${JOB_ID}`);
        const data     = await response.json();

        // HTTP 오류 (작업 자체가 없을 때만)
        if (response.status === 404) {
            showError('작업을 찾을 수 없습니다');
            stopPolling();
            return;
        }

        // 진행률 업데이트
        updateProgress(data.progress, data.stage, data.stage_idx);

        // 분석 완료
        if (data.status === 'done') {
            updateProgress(100, '분석 완료!', 5);
            stopPolling();
            setTimeout(() => {
                window.location.href = `/result/${JOB_ID}`;
            }, 800);
            return;
        }

        // 분석 오류 (status가 명시적으로 error일 때만)
        if (data.status === 'error') {
            showError(data.error || '분석 중 오류가 발생했습니다');
            stopPolling();
            return;
        }

        // 그 외 (pending / running) → 계속 진행
    } catch (err) {
        console.error('폴링 오류:', err);
    }
}

// ── 진행률 업데이트 ──────────────────────────────────────
function updateProgress(percent, stage, stageIdx) {
    progressFill.style.width = percent + '%';
    percentText.textContent  = percent + '%';
    stageText.textContent    = stage;

    // 단계별 활성화 상태 업데이트
    stageItems.forEach((item) => {
        const stageNum = parseInt(item.dataset.stage);
        item.classList.remove('active', 'completed');
        if (stageNum < stageIdx) {
            item.classList.add('completed');
        } else if (stageNum === stageIdx) {
            item.classList.add('active');
        }
    });
}

// ── 오류 표시 ────────────────────────────────────────────
function showError(message) {
    errorMsg.hidden        = false;
    errorMsg.style.display = 'flex';
    errorDetail.textContent = message;
}

// ── 폴링 시작/종료 ───────────────────────────────────────
function startPolling() {
    pollStatus();
    pollInterval = setInterval(pollStatus, 1500);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// ── 시작 ─────────────────────────────────────────────────
startPolling();