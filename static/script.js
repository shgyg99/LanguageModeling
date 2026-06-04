let timeoutId = null;
let lastWord = '';
const promptInput = document.getElementById('prompt');
const suggestionsList = document.getElementById('suggestionsList');
const temperatureSlider = document.getElementById('temperature');
const tempValue = document.getElementById('tempValue');
const topKSlider = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');

// Update displayed values
temperatureSlider.addEventListener('input', () => {
    tempValue.textContent = temperatureSlider.value;
});

topKSlider.addEventListener('input', () => {
    topKValue.textContent = topKSlider.value;
});

// Fetch suggestions from backend
async function fetchSuggestions() {
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Type something then press Space to see suggestions...</div>';
        return;
    }

    suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spinner"></i> Predicting next word...</div>';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                temperature: parseFloat(temperatureSlider.value),
                top_k: parseInt(topKSlider.value)
            })
        });

        const data = await response.json();
        
        if (data.error) {
            suggestionsList.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-triangle"></i> ${data.error}</div>`;
        } else if (data.suggestions && data.suggestions.length > 0) {
            displaySuggestions(data.suggestions);
        } else {
            suggestionsList.innerHTML = '<div class="no-suggestions"><i class="fas fa-lightbulb"></i> No suggestions found. Keep writing!</div>';
        }
    } catch (error) {
        console.error('Error:', error);
        suggestionsList.innerHTML = '<div class="error-message"><i class="fas fa-exclamation-circle"></i> Connection error. Please try again.</div>';
    }
}

// Display suggestions in the list
function displaySuggestions(suggestions) {
    suggestionsList.innerHTML = '';
    
    suggestions.forEach((sugg, index) => {
        const p = document.createElement('p');
        p.className = 'suggestion-item';
        
        const percent = Math.round(sugg.probability * 100);
        
        p.innerHTML = `
            <i class="fas fa-arrow-right" style="margin-right: 10px; color: #71a8ac;"></i>
            <strong>${escapeHtml(sugg.token)}</strong>
            <span style="float: right; font-size: 0.85rem; color: #ee865d;">${percent}%</span>
        `;
        
        p.onclick = () => {
            const currentText = promptInput.value;
            const space = currentText.endsWith(' ') ? '' : ' ';
            const newText = currentText + space + sugg.token;
            promptInput.value = newText;
            suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Press Space to see next suggestions...</div>';
            promptInput.focus();
        };
        
        suggestionsList.appendChild(p);
    });
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle input events - detect space on mobile
let lastValue = '';

promptInput.addEventListener('input', (e) => {
    const currentValue = promptInput.value;
    
    // Check if a space was added (for mobile)
    if (currentValue.length > lastValue.length && currentValue[lastValue.length] === ' ') {
        console.log("⭐ Space detected via input event");
        fetchSuggestions();
    }
    
    lastValue = currentValue;
});

// Handle keydown for desktop browsers
promptInput.addEventListener('keydown', (e) => {
    // Space key detection for desktop
    if (e.key === ' ' || e.key === 'Space' || e.code === 'Space') {
        e.preventDefault();
        
        const cursorPos = promptInput.selectionStart;
        const text = promptInput.value;
        const newText = text.slice(0, cursorPos) + ' ' + text.slice(cursorPos);
        promptInput.value = newText;
        promptInput.selectionStart = promptInput.selectionEnd = cursorPos + 1;
        lastValue = newText;
        
        fetchSuggestions();
    }
    
    // Enter key
    if (e.key === 'Enter') {
        e.preventDefault();
        fetchSuggestions();
    }
    
    // Escape key
    if (e.key === 'Escape') {
        suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Suggestions cleared. Press Space to get new ones.</div>';
    }
});

// Also listen for beforeinput to catch space on some mobile browsers
promptInput.addEventListener('beforeinput', (e) => {
    if (e.data === ' ') {
        setTimeout(() => {
            fetchSuggestions();
        }, 10);
    }
});

// Handle composition (IME for non-Latin input)
let composing = false;
promptInput.addEventListener('compositionstart', () => {
    composing = true;
});
promptInput.addEventListener('compositionend', () => {
    composing = false;
});

// Initial message
suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Type something then press Space to see suggestions.</div>';