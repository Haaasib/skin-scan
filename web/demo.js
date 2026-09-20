const API_URL = window.location.origin;

const fileInput = document.getElementById('fileInput');
const apiKeyInput = document.getElementById('apiKeyInput');
const scanBtn = document.getElementById('scanBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');
const scoresContainer = document.getElementById('scores');
const overlaysContainer = document.getElementById('overlays');
const issuesContainer = document.getElementById('issues');
const tagsContainer = document.getElementById('tags');
const profileEl = document.getElementById('profile');

let selectedFile = null;
let originalImage = null;

function refreshScanEnabled() {
    const ready = Boolean(selectedFile) && Boolean(apiKeyInput.value.trim());
    scanBtn.disabled = !ready;
    scanBtn.style.opacity = ready ? '1' : '0.5';
    scanBtn.style.cursor = ready ? 'pointer' : 'not-allowed';
}

apiKeyInput.addEventListener('input', refreshScanEnabled);

fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];

    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            originalImage = new Image();
            originalImage.src = ev.target.result;
        };
        reader.readAsDataURL(selectedFile);
    } else {
        originalImage = null;
    }
    refreshScanEnabled();
});

scanBtn.addEventListener('click', async () => {
    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        alert('Enter your API key');
        return;
    }
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    error.style.display = 'none';
    results.classList.remove('active');
    loading.style.display = 'block';
    scanBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: {
                'X-API-Key': apiKey,
            },
            body: formData,
        });

        if (!response.ok) {
            let detail = 'Scan failed';
            try {
                const errorData = await response.json();
                detail = errorData.detail || detail;
            } catch (_) {}
            throw new Error(detail);
        }

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        error.textContent = `Error: ${err.message}`;
        error.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        refreshScanEnabled();
    }
});

function displayResults(data) {
    const profile = data.profile || {};
    profileEl.textContent = `Skin type: ${(profile.skin_type || 'unknown')} · Models: ${(profile.models_used || []).join(', ') || 'cv-only'} · Top: ${(profile.top_issues || []).join(', ') || 'none'}`;

    issuesContainer.innerHTML = '';
    const issues = data.issues || [];
    if (issues.length === 0) {
        issuesContainer.innerHTML = '<p>No major issues flagged.</p>';
    } else {
        for (const issue of issues) {
            const card = document.createElement('div');
            card.className = `issue-card ${issue.severity}`;
            card.innerHTML = `
                <h4>${issue.label} · ${issue.severity.toUpperCase()} · ${(issue.score * 100).toFixed(0)}%</h4>
                <p>${issue.summary}</p>
            `;
            issuesContainer.appendChild(card);
        }
    }

    tagsContainer.innerHTML = '';
    for (const tag of (data.concern_tags || [])) {
        const el = document.createElement('span');
        el.className = 'tag';
        el.textContent = tag;
        tagsContainer.appendChild(el);
    }

    scoresContainer.innerHTML = '';
    for (const [category, score] of Object.entries(data.scores)) {
        const card = document.createElement('div');
        card.className = 'score-card';
        card.innerHTML = `
            <h3>${category}</h3>
            <div class="score-value">${(score * 100).toFixed(0)}%</div>
        `;
        scoresContainer.appendChild(card);
    }

    overlaysContainer.innerHTML = '';
    for (const [category, overlayDataURL] of Object.entries(data.overlays)) {
        const card = document.createElement('div');
        card.className = 'overlay-card';

        const title = document.createElement('h4');
        title.textContent = category;
        card.appendChild(title);

        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'canvas-container';

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const overlay = new Image();
        overlay.onload = () => {
            if (originalImage && originalImage.complete) {
                canvas.width = originalImage.width;
                canvas.height = originalImage.height;
                ctx.drawImage(originalImage, 0, 0);
                ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
            } else {
                canvas.width = overlay.width;
                canvas.height = overlay.height;
                ctx.drawImage(overlay, 0, 0);
            }
        };
        overlay.src = overlayDataURL;

        canvasContainer.appendChild(canvas);
        card.appendChild(canvasContainer);
        overlaysContainer.appendChild(card);
    }

    results.classList.add('active');
}

refreshScanEnabled();
