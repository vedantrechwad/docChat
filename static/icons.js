/** Material Symbols icon helpers for CarnetLM */
function icon(name, size = 20) {
    return `<span class="material-symbols-outlined" style="font-size:${size}px">${name}</span>`;
}

function refreshIcons(root) {
    // Material Symbols are a web font — no JS initialization needed.
    // This function is kept for backward compatibility with existing calls.
}

function sourceIcon(type) {
    const map = {
        website: 'language',
        youtube: 'play_circle',
        clipboard: 'content_paste',
        document: 'description',
        file: 'description',
    };
    const norm = (type || '').toLowerCase().trim();
    if (norm.includes('web') || norm.includes('link') || norm.includes('url')) return 'language';
    if (norm.includes('youtube') || norm.includes('video') || norm.includes('play')) return 'play_circle';
    if (norm.includes('clip') || norm.includes('paste')) return 'content_paste';
    return map[norm] || 'description';
}

