const API_URL = window.location.origin;

const fileInput = document.getElementById('fileInput');
const scanBtn = document.getElementById('scanBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');
const scoresContainer = document.getElementById('scores');
const overlaysContainer = document.getElementById('overlays');
const issuesContainer = document.getElementById('issues');
const tagsContainer = document.getElementById('tags');

let selectedFile = null;
let originalImage = null;

// Enable scan button when file is selected
fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];

    if (selectedFile) {
        scanBtn.disabled = false;
        scanBtn.style.opacity = '1';
        scanBtn.style.cursor = 'pointer';

        // Load original image
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage = new Image();
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(selectedFile);
    } else {
        scanBtn.disabled = true;
        scanBtn.style.opacity = '0.5';
        scanBtn.style.cursor = 'not-allowed';
    }
});

// Scan button click handler
scanBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    // Reset UI
    error.style.display = 'none';
    results.classList.remove('active');
    loading.style.display = 'block';
    scanBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Call API
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Scan failed');
        }

        const data = await response.json();

        // Display results
        displayResults(data);

    } catch (err) {
        error.textContent = `Error: ${err.message}`;
        error.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        scanBtn.disabled = false;
    }
});

function displayResults(data) {
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

    // Display overlays
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

        // Load and composite overlay
        const overlay = new Image();
        overlay.onload = () => {
            // Set canvas size to match original image
            if (originalImage && originalImage.complete) {
                canvas.width = originalImage.width;
                canvas.height = originalImage.height;

                // Draw original image
                ctx.drawImage(originalImage, 0, 0);

                // Draw overlay
                ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
            } else {
                // Fallback: just show overlay
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
