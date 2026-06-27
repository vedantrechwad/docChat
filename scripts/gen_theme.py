import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "static" / "index.html"
content = path.read_text(encoding="utf-8")
m = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
css = m.group(1)

replacements = [
    (":root {", ":root {\n            --bg-elevated: #2a2a2e;\n            --shadow-elevated: 0 1px 0 rgba(255,255,255,0.05) inset, 0 12px 40px rgba(0,0,0,0.35);"),
    ("--black: #000000;", "--black: #121214;"),
    ("--white: #ffffff;", "--white: #f0eeea;"),
    ("--g05: #0d0d0d;", "--g05: #161618;"),
    ("--g10: #1a1a1a;", "--g10: #1e1e21;"),
    ("--g15: #262626;", "--g15: #252528;"),
    ("--g20: #333333;", "--g20: #2e2e32;"),
    ("--g90: #e6e6e6;", "--g90: #eceae6;"),
    ("--radius: 2px;", "--radius: 6px;"),
    ("--radius-sm: 1px;", "--radius-sm: 4px;"),
    ("--radius-lg: 2px;", "--radius-lg: 10px;"),
    ("--shadow: 0 2px 24px rgba(0,0,0,0.8);", "--shadow: var(--shadow-elevated);"),
]
for old, new in replacements:
    css = css.replace(old, new)

css = css.replace(
    "rgba(0,0,0,0.03) 2px,\n                rgba(0,0,0,0.03) 4px",
    "rgba(0,0,0,0.015) 2px,\n                rgba(0,0,0,0.015) 4px",
)
css = css.replace(
    "border-color: var(--g25, #404040);",
    "border-color: rgba(255,255,255,0.08); opacity: 0.15;",
)
css = css.replace(
    "background: rgba(10,10,10,0.92);",
    "background: var(--g05);\n            box-shadow: 4px 0 24px rgba(0,0,0,0.25);",
)
css = css.replace(
    """.logo::before {
            content: '▸';
            color: var(--g50);
            font-size: 0.7rem;
        }""",
    ".logo-icon { color: var(--g50); flex-shrink: 0; }",
)
css = css.replace(
    ".main-header {",
    ".main-header {\n            padding: 0.5rem 1rem 0;\n            gap: 0.25rem;",
)
css = css.replace(
    ".tab {\n            padding: 0.875rem 1.25rem;",
    ".tab {\n            padding: 0.5rem 1rem;",
)
css = css.replace(
    "border-bottom: 1px solid transparent;",
    "border-radius: var(--radius);",
)
css = css.replace(
    ".tab.active { color: var(--white); border-bottom-color: var(--white); }",
    ".tab.active { color: var(--text-primary); background: var(--g15); box-shadow: var(--shadow-elevated); }",
)
css = css.replace(
    """.main-header::after {
            content: 'CARNETLM // OFFLINE AI NOTEBOOK';
            position: absolute;
            right: 1rem;
            font-family: var(--font-mono);
            font-size: 0.4375rem;
            color: var(--g25, #404040);
            letter-spacing: 0.1em;
            pointer-events: none;
        }""",
    "",
)
css = css.replace(
    "border-radius: 0;\n            line-height: 1.65;",
    "border-radius: var(--radius);\n            line-height: 1.65;",
)
css = css.replace(
    "clip-path: polygon(0 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%);",
    "",
)
css = css.replace("border-bottom-right-radius: 0;", "")
css = css.replace(
    "border-bottom-left-radius: 0;\n            border-left: 2px solid var(--g30);\n            clip-path: polygon(0 0, 100% 0, 100% 100%, 6px 100%, 0 calc(100% - 6px));",
    "border-left: 3px solid var(--g30);",
)
css = css.replace(
    "clip-path: polygon(0 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%);",
    "border-radius: var(--radius);",
)
css = css.replace(
    ".send-btn {\n            width: 32px; height: 32px;\n            border-radius: 0;",
    ".send-btn {\n            width: 32px; height: 32px;\n            border-radius: var(--radius-sm);",
)
css = css.replace("background: rgba(0,0,0,0.85);", "background: rgba(18,18,20,0.72);")
css = css.replace(
    ".modal {\n            background: var(--g10);\n            border: 1px solid var(--g25, #404040);\n            border-radius: 0;",
    ".modal {\n            background: var(--g10);\n            border: 1px solid var(--g20);\n            border-radius: var(--radius-lg);",
)
css = css.replace(
    ".btn-primary { background: var(--white);",
    ".btn-ghost { background: transparent; border-color: transparent; }\n        .btn-ghost:hover { background: var(--g10); border-color: var(--g20); }\n        .btn-soft { background: var(--g15); }\n        .btn-primary { background: var(--white); box-shadow: 0 1px 2px rgba(0,0,0,0.2);",
)
css = css.replace(
    ".src-icon {\n            width: 18px; height: 18px;\n            border-radius: 0;",
    ".src-icon {\n            width: 22px; height: 22px;\n            border-radius: var(--radius-sm);\n            background: var(--bg-elevated);",
)
css = css.replace("text-transform: uppercase;", "text-transform: none;")

css += """

        .icon { display: inline-flex; vertical-align: middle; stroke: currentColor; fill: none; flex-shrink: 0; }
        .sidebar-add .icon { opacity: 0.7; }

        .onboarding-card {
            margin: 1rem 2rem;
            padding: 1.25rem 1.5rem;
            background: var(--g10);
            border: 1px solid var(--g20);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-elevated);
        }
        .onboarding-card h4 { font-size: 0.875rem; margin-bottom: 0.75rem; color: var(--text-primary); }
        .onboarding-steps { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
        .onboarding-step { display: flex; align-items: center; gap: 0.625rem; font-size: 0.75rem; color: var(--g60); }
        .onboarding-step-num { width: 22px; height: 22px; border-radius: 50%; background: var(--g15); display: flex; align-items: center; justify-content: center; font-size: 0.625rem; font-family: var(--font-mono); color: var(--g50); }
        .settings-section { margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid var(--g15); }
        .settings-section:last-child { border-bottom: none; margin-bottom: 0; }
        .settings-section h4 { font-size: 0.6875rem; color: var(--g50); margin-bottom: 0.625rem; font-family: var(--font-mono); letter-spacing: 0.06em; }
        .health-dots { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .health-dot { display: flex; align-items: center; gap: 0.375rem; font-size: 0.6875rem; color: var(--g60); padding: 0.35rem 0.625rem; background: var(--g05); border-radius: var(--radius-sm); border: 1px solid var(--g15); }
        .health-dot .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--g30); }
        .health-dot.ok .dot { background: #6b9e6b; }
        .health-dot.bad .dot { background: var(--g40); }
        .chunk-highlight { background: rgba(240,238,234,0.08); outline: 1px solid var(--g30); border-radius: var(--radius-sm); padding: 0.25rem; }
        .stream-cursor { display: inline-block; width: 2px; height: 1em; background: var(--g50); animation: blink 1s step-end infinite; vertical-align: text-bottom; margin-left: 1px; }
        @keyframes blink { 50% { opacity: 0; } }
        .ql-toolbar.ql-snow { background: var(--g10) !important; border-color: var(--g20) !important; border-radius: var(--radius) var(--radius) 0 0; }
        .ql-container.ql-snow { background: var(--g05) !important; border-color: var(--g20) !important; border-radius: 0 0 var(--radius) var(--radius); }
        .ql-snow .ql-stroke { stroke: var(--g50) !important; }
        .ql-snow .ql-fill { fill: var(--g50) !important; }
        .ql-snow .ql-picker-label { color: var(--g50) !important; }
        .ql-snow .ql-picker-options { background: var(--g10) !important; border-color: var(--g20) !important; }
        .ql-snow .ql-picker-item { color: var(--g60) !important; }
        .ql-snow .ql-picker-item:hover { color: var(--white) !important; }
        .ql-editor { color: var(--g80) !important; }
        .ql-editor.ql-blank::before { color: var(--g30) !important; font-style: normal !important; }
        .sidebar-actions { padding: 0.25rem 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }
        .sidebar-actions-row { display: flex; gap: 0.5rem; flex-wrap: wrap; width: 100%; }
        .settings-btn-wrap { padding: 0.5rem 0.75rem; border-top: 1px solid var(--g15); }
"""

out = Path(__file__).resolve().parents[1] / "static" / "theme.css"
out.write_text("/* CarnetLM matte minimal theme */\n" + css.strip() + "\n", encoding="utf-8")
print("Wrote", out, len(css), "chars")
