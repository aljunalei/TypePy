import time
import random
import re
import html
from io import StringIO
import tokenize

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Salesforce/codegen-350M-multi"

FALLBACK_SNIPPETS = [
    """def greet(name: str) -> str:
    return f"Hello, {name}!"
print(greet("World"))""",
    """nums = [1, 2, 3, 4, 5]
squares = [n * n for n in nums if n % 2 == 1]
print(squares)""",
    """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
print(factorial(5))""",
]

PROMPT_TEMPLATES = {
    "fundamentals":
        "Write a short Python code snippet (5-12 lines) demonstrating {topic}. "
        "Keep it self-contained, clean, and runnable. Avoid external I/O unless necessary. "
        "Output only code with no comments or explanations.",
    "intermediate":
        "Write a short Python code snippet (6-14 lines) demonstrating {topic}. "
        "Prefer clarity and idiomatic Python. Keep it self-contained. "
        "Output only code with no comments or explanations.",
    "advanced":
        "Write a concise Python example (6-14 lines) demonstrating {topic}. "
        "Use clear variable names and include a minimal demo. "
        "Output only code with no comments or explanations."
}

TOPICS = [
    "list comprehensions",
    "dictionary usage",
    "function definition with type hints",
    "class with __init__ and method",
    "file reading and writing",
    "decorators",
    "generators and yield",
    "context managers",
    "error handling with try/except",
    "sorting with key functions",
    "lambda functions and map/filter",
]

# -----------------------------
# Model loading
# -----------------------------
tokenizer = None
model = None
model_load_error = None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except Exception as e:
    model_load_error = str(e)

# -----------------------------
# Snippet generation
# -----------------------------
def _strip_comment_only_lines(code: str) -> str:
    lines = code.splitlines()
    # Trim leading comment/empty lines
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
        lines.pop(0)
    # Trim trailing comment/empty lines
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
        lines.pop()
    return "\n".join(lines)

def generate_snippet(difficulty: str, topic: str, seed: int | None = None) -> str:
    random.seed(seed)
    if model is None:
        return random.choice(FALLBACK_SNIPPETS)

    key = ("fundamentals" if difficulty == "Easy"
           else "intermediate" if difficulty == "Medium"
           else "advanced")
    prompt = PROMPT_TEMPLATES[key].format(topic=topic)
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Extract fenced code or fallback
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        code = blocks[0].strip()
    else:
        parts = text.split("\n", 1)
        code = parts[1] if len(parts) > 1 else text
        lines = code.splitlines()
        filtered = []
        for ln in lines:
            if len(filtered) == 0 and ln.strip().startswith("#"):
                continue
            filtered.append(ln)
        code = "\n".join(filtered).strip()

    code = _strip_comment_only_lines(code)

    if not code or code.count("\n") < 3 or code.count("\n") > 30:
        return random.choice(FALLBACK_SNIPPETS)
    return code

# -----------------------------
# Masking logic
# -----------------------------
PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
}

def select_mask_positions(code: str, difficulty: str, rng: random.Random):
    reader = StringIO(code).readline
    maskables = []
    try:
        for tok in tokenize.generate_tokens(reader):
            ttype, tstr, (sr, sc), (er, ec), _ = tok
            if ttype == tokenize.NAME and tstr not in PY_KEYWORDS:
                maskables.append((sr, sc, er, ec, tstr, "name"))
            elif ttype == tokenize.NAME:
                maskables.append((sr, sc, er, ec, tstr, "keyword"))
            elif ttype == tokenize.NUMBER:
                maskables.append((sr, sc, er, ec, tstr, "number"))
            elif ttype == tokenize.STRING and len(tstr) <= 18:
                maskables.append((sr, sc, er, ec, tstr, "string"))
    except tokenize.TokenError:
        words = re.finditer(r"\b[A-Za-z_][A-Za-z_0-9]*\b", code)
        return [{"start": m.start(), "end": m.end(), "text": m.group()}
                for m in words][:6]

    weights = []
    for *_, text, kind in maskables:
        if kind == "name":
            w = 1.0
        elif kind == "keyword":
            w = (0.3 if difficulty == "Easy"
                 else 0.6 if difficulty == "Medium"
                 else 0.9)
        elif kind == "number":
            w = 0.6
        else:
            w = 0.5
        weights.append(max(w, 0.05))

    if not maskables:
        return []
    base = 4 if difficulty == "Easy" else (6 if difficulty == "Medium" else 8)
    count = min(base, len(maskables))

    chosen = []
    avail = list(range(len(maskables)))
    for _ in range(count):
        total = sum(weights[i] for i in avail)
        pick = rng.random() * total
        cum = 0
        for i in avail:
            cum += weights[i]
            if cum >= pick:
                chosen.append(i)
                avail.remove(i)
                break

    # Convert row/col to absolute offsets
    lines = code.splitlines(keepends=True)
    offsets = []
    cur = 0
    for line in lines:
        offsets.append(cur)
        cur += len(line)

    def to_abs(sr, sc, er, ec):
        return offsets[sr - 1] + sc, offsets[er - 1] + ec

    spans = []
    for idx in chosen:
        sr, sc, er, ec, txt, _ = maskables[idx]
        s, e = to_abs(sr, sc, er, ec)
        spans.append({"start": s, "end": e, "text": txt})
    spans.sort(key=lambda x: x["start"])
    return spans

def apply_masks(code: str, spans: list[dict]):
    masked, last, answers = [], 0, []
    for i, sp in enumerate(spans, 1):
        s, e, txt = sp["start"], sp["end"], sp["text"]
        normal = code[last:s]
        masked.append(html.escape(normal))
        placeholder = f"__[{i}]__"
        masked.append(
            f"<span class='placeholder'>{html.escape(placeholder)}</span>"
        )
        answers.append(txt)
        last = e
    masked.append(html.escape(code[last:]))
    return "".join(masked), answers

# -----------------------------
# Theme CSS (Single Vibrant Yellow Theme)
# -----------------------------
vibrant_css = """
<style>
:root {
  --bg: #FFFBE5;               /* Soft off-white background */
  --fg: #2F253A;               /* Neutral dark text */
  --border: #E6C229;           /* Warm yellow-gold border */
  --placeholder-bg: #FEF0A3;   /* Light yellow placeholder bg */
  --placeholder-fg: #2F253A;   /* Dark text for placeholders */
  --placeholder-border: #E6C229;
  --accent: #FEDC2A;           /* Primary vibrant yellow accent */
  --button-bg: var(--accent);
  --button-fg: #2F253A;        /* Dark button text */
  --button-hover: #FFD93D;     /* Slightly brighter yellow */
  --legend-fg: #902AFE;        /* Vibrant purple for legends */
  --generated-text: #330069;   /* Deep purple for code text */
}
.instruction, .blanks-label {
  color: var(--fg);
  font-weight: bold;
}
.gradio-container {
  background: var(--bg);
  color: var(--fg);
}
.gr-button, button {
  background: var(--button-bg) !important;
  color: var(--button-fg) !important;
}
.gr-button:hover, button:hover {
  background: var(--button-hover) !important;
}
.codebox {
  background: var(--placeholder-bg);
  color: var(--generated-text);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  font-family: ui-monospace, Menlo, Consolas, monospace;
  white-space: pre;
  overflow: auto;
  line-height: 1.5;
  font-size: 15px;
}
.placeholder {
  color: var(--placeholder-fg);
  background: var(--placeholder-bg);
  padding: 0 3px;
  border-radius: 4px;
  border: 1px dashed var(--placeholder-border);
  font-weight: 600;
}
.legend {
  color: var(--legend-fg);
  font-size: 13px;
  margin: 6px 0 0;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
</style>
"""

def get_instruction_html():
    return (
        "<div class='instruction'>Type the missing parts in order. "
        "Separate by spaces or new lines, then Submit.</div>"
    )

# -----------------------------
# Game logic
# -----------------------------
def start_new_game(difficulty, topic, seed):
    rng = random.Random(seed if seed else random.randrange(1_000_000_000))
    snippet = generate_snippet(difficulty, topic, seed=rng.randrange(1_000_000_000))
    spans = select_mask_positions(snippet, difficulty, rng)
    if not spans:
        # fallback masks
        words = list(re.finditer(r"\b[A-Za-z_]\w*\b", snippet))
        random.shuffle(words)
        spans = [{"start": m.start(), "end": m.end(), "text": m.group()}
                 for m in sorted(words[:4], key=lambda x: x.start())]
    masked_html, answers = apply_masks(snippet, spans)
    hint_text = ", ".join([f"[{i+1}] ({len(ans)} chars)"
                           for i, ans in enumerate(answers)])
    blanks_color = "var(--fg)"
    html_code = (
        f"<div class='codebox'>{masked_html}</div>"
        f"<div class='legend'><span class='blanks-label' style='color:{blanks_color}; font-weight:bold;'>Blanks:</span> {html.escape(hint_text)}</div>"
    )

    meta = {
        "answers": answers,
        "start_time": time.time(),
    }
    return html_code, hint_text, meta, ""  # clear input

def score_attempt(user_input, meta):
    if not meta or "answers" not in meta:
        return "Start a new game first.", gr.update(visible=True)
    answers = meta["answers"]
    elapsed = max(0.01, time.time() - meta["start_time"])
    tokens = [t for t in re.split(r"[\n,;]+|\s{2,}", user_input.strip()) if t]
    if len(tokens) < len(answers):
        tokens = user_input.strip().split()
    tokens = (tokens + [""] * len(answers))[:len(answers)]
    correct_flags = [u == a for u, a in zip(tokens, answers)]
    correct = sum(correct_flags)
    base = max(0, 10 * correct - 2 * sum(not ok for ok in correct_flags))
    cps = sum(len(a) for a in answers) / elapsed
    bonus = int(cps * 2.0 * (correct / max(1, len(answers))))
    total = base + bonus
    accuracy = 100 * correct / max(1, len(answers))
    feedback = (
        f"Score: {total} | Correct: {correct}/{len(answers)} | "
        f"Time: {elapsed:.1f}s | Accuracy: {accuracy:.1f}%\n"
        f"Speed: {cps:.2f} chars/s\n\n" +
        "\n".join(
            f"[{i+1}] {'✓' if ok else '✗'} Your: {tokens[i] or '∅'}  "
            f"Expected: {answers[i]}"
            for i, ok in enumerate(correct_flags)
        )
    )
    return feedback, gr.update(visible=True)

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="PyType Snippets (Vibrant Yellow Theme)") as demo:
    # Apply single theme globally
    gr.HTML(vibrant_css)
    
    gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                    <h1>🎮TypePy</h1>
                    <h4>v1.0 - Typing and Analyzing Python Snippets</h4>
                    <h5 style="margin: 0;">The model will generate a Python code snippet based on your choice! You must analyze the type the missing values</h5>
                </div>
            </div>
            """
           )

    

    # Controls
    with gr.Row():
        with gr.Column(scale=3):
            difficulty = gr.Radio(["Easy", "Medium", "Hard"],
                                  value="Medium",
                                  label="Difficulty")
            topic = gr.Dropdown(TOPICS,
                                value="list comprehensions",
                                label="Topic")
            seed = gr.Number(value=None,
                             label="Seed (optional)",
                             precision=0)
            start_btn = gr.Button("New Snippet", variant="primary")
            hint_box = gr.Textbox(label="Hints",
                                  interactive=False)
            if model_load_error:
                gr.Markdown(f"> Using fallback snippets. "
                            f"Load error: {html.escape(model_load_error)[:200]}")
        with gr.Column(scale=7):
            code_html = gr.HTML()
            instruction_html = gr.HTML(get_instruction_html())
            user_answers = gr.Textbox(lines=4,
                                      placeholder="Answer for [1], then [2], ...")
            submit_btn = gr.Button("Submit")
            feedback_box = gr.Textbox(label="Result",
                                      interactive=False)

    meta_state = gr.State()

    # Event bindings
    start_btn.click(fn=start_new_game,
                    inputs=[difficulty, topic, seed],
                    outputs=[code_html, hint_box, meta_state, user_answers])

    submit_btn.click(fn=score_attempt,
                     inputs=[user_answers, meta_state],
                     outputs=[feedback_box, feedback_box])

if __name__ == "__main__":
    demo.launch()