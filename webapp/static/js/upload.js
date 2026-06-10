/* ═══════════════════════════════════════════════════
   AI-CASA Upload Script
   ─ 영상 파일 선택 / 드래그앤드롭 / 분석 시작
═══════════════════════════════════════════════════ */

const dropzone   = document.getElementById('dropzone');
const fileInput  = document.getElementById('file-input');
const fileInfo   = document.getElementById('file-info');
const fileName   = document.getElementById('file-name');
const fileSize   = document.getElementById('file-size');
const fileRemove = document.getElementById('file-remove');
const btnAnalyze = document.getElementById('btn-analyze');

let selectedFile = null;

// ── 파일 선택 (클릭) ─────────────────────────────────────
dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// ── 드래그앤드롭 ─────────────────────────────────────────
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-active');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-active');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-active');

    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// ── 파일 처리 ────────────────────────────────────────────
function handleFile(file) {
    // 형식 검증
    const validTypes = ['video/mp4', 'video/avi',
                        'video/quicktime', 'video/x-msvideo'];
    const validExts  = ['.mp4', '.avi', '.mov'];
    const ext = file.name.toLowerCase()
        .substring(file.name.lastIndexOf('.'));

    if (!validTypes.includes(file.type) &&
        !validExts.includes(ext)) {
        alert('⚠️ MP4, AVI, MOV 형식만 지원됩니다.');
        return;
    }

    // 크기 검증 (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('⚠️ 파일 크기는 500MB 이하여야 합니다.');
        return;
    }

    // 파일 정보 표시
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.hidden = false;
    btnAnalyze.disabled = false;
    dropzone.style.display = 'none';
}

// ── 파일 제거 ────────────────────────────────────────────
fileRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.hidden = true;
    btnAnalyze.disabled = true;
    dropzone.style.display = 'block';
});

// ── 파일 크기 포맷 ───────────────────────────────────────
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024)
        return (bytes / 1024 / 1024).toFixed(1) + ' MB';
    return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

// ── 분석 시작 (XHR 업로드 + 진행률 표시) ──────────────────
function showProgress(pct) {
    let bar = document.getElementById('upload-progress');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'upload-progress';
        bar.style.cssText =
            'margin-top:12px;height:8px;border-radius:6px;' +
            'background:#e5e7eb;overflow:hidden;';
        bar.innerHTML =
            '<div id="upload-progress-fill" style="height:100%;' +
            'width:0;background:#4f46e5;transition:width .15s;"></div>';
        btnAnalyze.insertAdjacentElement('afterend', bar);
    }
    document.getElementById('upload-progress-fill').style.width = pct + '%';
    btnAnalyze.innerHTML = pct < 100
        ? `⏳ 업로드 중... ${pct}%`
        : '⌛ 서버 처리 중...';
}

function resetAnalyzeButton(msg) {
    if (msg) alert('❌ 오류: ' + msg);
    btnAnalyze.disabled = false;
    btnAnalyze.innerHTML = '🔬 분석 시작하기';
    const bar = document.getElementById('upload-progress');
    if (bar) bar.remove();
}

btnAnalyze.addEventListener('click', () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    showProgress(0);

    const formData = new FormData();
    formData.append('video', selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/analyze');

    // 업로드 진행률 (실제 전송 바이트 기준)
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            showProgress(Math.round((e.loaded / e.total) * 100));
        }
    });

    // 서버 응답 수신
    xhr.addEventListener('load', () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300 && data.job_id) {
            showProgress(100);
            window.location.href = `/analyzing/${data.job_id}`;
        } else {
            resetAnalyzeButton(data.error || `업로드 실패 (HTTP ${xhr.status})`);
        }
    });

    xhr.addEventListener('error', () => resetAnalyzeButton('네트워크 오류'));
    xhr.addEventListener('abort', () => resetAnalyzeButton('업로드 취소됨'));

    xhr.send(formData);
});
