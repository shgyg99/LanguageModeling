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

// Get last word from text
function getLastWord(text) {
    const words = text.trim().split(/\s+/);
    return words[words.length - 1] || '';
}

// Fetch suggestions from backend
async function fetchSuggestions() {
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Press Space to see suggestions...</div>';
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
            // Add a space before the suggestion if the last character is not a space
            const space = currentText.endsWith(' ') ? '' : ' ';
            const newText = currentText + space + sugg.token;
            promptInput.value = newText;
            // Do NOT automatically fetch new suggestions – wait for user to press Space again
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

// Handle keydown: ONLY generate on Space or Enter
promptInput.addEventListener('keydown', (e) => {
    // Generate suggestions on Space
    if (e.key === ' ' || e.key === 'Space') {
        e.preventDefault();  // prevent default space insertion? We'll handle manually to avoid double space
        // Insert a space manually
        const cursorPos = promptInput.selectionStart;
        const text = promptInput.value;
        const newText = text.slice(0, cursorPos) + ' ' + text.slice(cursorPos);
        promptInput.value = newText;
        // Move cursor after the space
        promptInput.selectionStart = promptInput.selectionEnd = cursorPos + 1;
        
        // Now fetch suggestions
        fetchSuggestions();
    }
    
    // Generate suggestions on Enter as well (optional)
    if (e.key === 'Enter') {
        e.preventDefault();
        fetchSuggestions();
    }
    
    // Escape clears suggestions
    if (e.key === 'Escape') {
        suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Suggestions cleared. Press Space to get new ones.</div>';
    }
});

// Prevent any automatic fetching on input (typing letters, backspace, etc.)
promptInput.addEventListener('input', (e) => {
    // Do nothing – suggestions only on Space/Enter
    // But we can clear the "no suggestions" message if user keeps typing
    if (suggestionsList.innerHTML.includes('Press Space') === false && 
        suggestionsList.innerHTML.includes('cleared') === false) {
        // Optional: keep a neutral message
        suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Press Space to get suggestions.</div>';
    }
});

// Handle composition (IME) to avoid issues with non-Latin input
let composing = false;
promptInput.addEventListener('compositionstart', () => {
    composing = true;
});
promptInput.addEventListener('compositionend', () => {
    composing = false;
});

// Initial load – show a friendly message instead of fetching
suggestionsList.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Press Space to see suggestions.</div>';