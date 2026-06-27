"""Patch index.html for CarnetLM excellence UI."""
import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "static" / "index.html"
html = path.read_text(encoding="utf-8")

# Remove inline style block, add external assets
html = re.sub(
    r"\s*<style>.*?</style>",
    """
    <link rel="stylesheet" href="/static/theme.css">
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
    <script src="/static/icons.js"></script>""",
    html,
    count=1,
    flags=re.DOTALL,
)

# WebGL matte tint
html = html.replace(
    "float brightness = v * 0.13;\n            gl_FragColor = vec4(vec3(brightness), 1.0);",
    "float brightness = v * 0.08;\n            vec3 matte = vec3(0.102, 0.102, 0.114);\n            gl_FragColor = vec4(matte + vec3(brightness), 1.0);",
)

# Sidebar header
html = html.replace(
    '<div class="logo">CarnetLM</div>\n            <div class="logo-sub">LOCAL AI // OPTIONAL CLOUD</div>',
    '<div class="logo"><i data-lucide="book-open" class="logo-icon icon" style="width:16px;height:16px"></i> CarnetLM</div>',
)

# Sidebar buttons with icons
html = html.replace(
    """        <div style="padding:0.25rem 0.75rem; display:flex; gap:0.25rem; flex-wrap:wrap;">
            <button class="sidebar-add" onclick="openModal('upload')">▸ FILE</button>
            <button class="sidebar-add" onclick="openModal('url')">▸ URL</button>
            <button class="sidebar-add" onclick="openModal('youtube')">▸ YT</button>
            <button class="sidebar-add" onclick="openModal('clipboard')">▸ PASTE</button>
        </div>
        <div style="padding:0.25rem 0.75rem; display:flex; gap:0.25rem; flex-wrap:wrap;">
            <button class="sidebar-add" onclick="openModal('search')">◇ SEARCH</button>
            <button class="sidebar-add" onclick="runSummary()">◇ GUIDE</button>
            <button class="sidebar-add" onclick="openModal('compare')">◇ COMPARE</button>
        </div>""",
    """        <div class="sidebar-actions">
            <div class="sidebar-actions-row">
                <button class="sidebar-add" onclick="openModal('upload')"><i data-lucide="file-up" class="icon" style="width:14px;height:14px"></i> File</button>
                <button class="sidebar-add" onclick="openModal('url')"><i data-lucide="link" class="icon" style="width:14px;height:14px"></i> URL</button>
                <button class="sidebar-add" onclick="openModal('youtube')"><i data-lucide="youtube" class="icon" style="width:14px;height:14px"></i> YouTube</button>
                <button class="sidebar-add" onclick="openModal('clipboard')"><i data-lucide="clipboard-paste" class="icon" style="width:14px;height:14px"></i> Paste</button>
            </div>
            <div class="sidebar-actions-row">
                <button class="sidebar-add" onclick="openModal('search')"><i data-lucide="search" class="icon" style="width:14px;height:14px"></i> Search</button>
                <button class="sidebar-add" onclick="runSummary()"><i data-lucide="sparkles" class="icon" style="width:14px;height:14px"></i> Guide</button>
                <button class="sidebar-add" onclick="openModal('compare')"><i data-lucide="git-compare" class="icon" style="width:14px;height:14px"></i> Compare</button>
            </div>
        </div>
        <div class="settings-btn-wrap">
            <button class="sidebar-add" style="width:100%" onclick="openSettings()"><i data-lucide="settings" class="icon" style="width:14px;height:14px"></i> Settings</button>
        </div>""",
)

# Tabs sentence case
html = html.replace(
    '<button class="tab active" data-tab="chat" onclick="switchTab(\'chat\')">// CHAT</button>',
    '<button class="tab active" data-tab="chat" onclick="switchTab(\'chat\')">Chat</button>',
)
html = html.replace(
    '<button class="tab" data-tab="notes" onclick="switchTab(\'notes\')">// NOTES',
    '<button class="tab" data-tab="notes" onclick="switchTab(\'notes\')">Notes',
)
html = html.replace(
    '<button class="tab" data-tab="editor" onclick="switchTab(\'editor\')">// EDITOR</button>',
    '<button class="tab" data-tab="editor" onclick="switchTab(\'editor\')">Editor</button>',
)

# Chat empty + send button
html = html.replace(
    '<div class="chat-empty-icon">◈</div>',
    '<div class="chat-empty-icon"><i data-lucide="messages-square" style="width:32px;height:32px;opacity:0.2"></i></div>',
)
html = html.replace(
    '<button type="submit" class="send-btn" id="sendBtn">▶</button>',
    '<button type="submit" class="send-btn" id="sendBtn"><i data-lucide="send" style="width:16px;height:16px"></i></button>',
)

# Settings modal before Quill script
settings_modal = """
<div class="modal-overlay" id="modal-settings">
    <div class="modal" style="max-width:520px">
        <h3><i data-lucide="settings" class="icon" style="width:16px;height:16px"></i> Settings <button class="modal-close" onclick="closeModals()">&times;</button></h3>
        <div class="settings-section">
            <h4>Model</h4>
            <div id="settingsModelInfo" style="font-size:0.75rem;color:var(--g60)">Loading...</div>
        </div>
        <div class="settings-section">
            <h4>Chunk preset</h4>
            <select class="input" id="settingsChunkPreset">
                <option value="auto">Auto (recommended)</option>
                <option value="compact">Compact</option>
                <option value="balanced">Balanced</option>
                <option value="dense">Dense</option>
            </select>
            <button class="btn btn-soft btn-sm" style="margin-top:0.5rem" onclick="saveChunkPreset()">Save preset</button>
        </div>
        <div class="settings-section">
            <h4>Text-to-speech</h4>
            <div id="settingsTtsInfo" style="font-size:0.75rem;color:var(--g60)">Checking...</div>
        </div>
        <div class="settings-section">
            <h4>System health</h4>
            <div class="health-dots" id="settingsHealth"></div>
        </div>
    </div>
</div>

"""
html = html.replace("<!-- Quill -->", settings_modal + "<!-- Quill -->")

# Onboarding placeholder in chat panel
html = html.replace(
    '<div class="chat-messages" id="chatMessages">',
    '<div id="onboardingWrap"></div>\n                <div class="chat-messages" id="chatMessages">',
)

path.write_text(html, encoding="utf-8")
print("Patched HTML structure")
