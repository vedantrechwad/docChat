/** Lucide icon helpers for CarnetLM */
function icon(name, size = 16) {
    return `<i data-lucide="${name}" class="icon" style="width:${size}px;height:${size}px"></i>`;
}

function refreshIcons(root) {
    if (typeof lucide !== 'undefined') {
        lucide.createIcons({ attrs: { 'stroke-width': 1.75 }, nameAttr: 'data-lucide', root: root || document });
    }
}

function sourceIcon(type) {
    const map = {
        Website: 'globe',
        YouTube: 'play-circle',
        Clipboard: 'clipboard',
        Document: 'file-text',
    };
    return map[type] || 'file-text';
}
