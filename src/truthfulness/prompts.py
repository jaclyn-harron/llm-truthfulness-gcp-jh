"""Prompt construction shared by the zero-shot, fine-tuned, and explainer paths.

Keeping the *user* message identical across predictors is deliberate: the
fine-tuned model must be a drop-in replacement for the zero-shot one, so they
must see the same rendered statement. Only the system message and the expected
output format differ.
"""

from __future__ import annotations

# Field label -> human-readable prefix in the rendered point.
_FIELD_ORDER = [
    ("statement", "Statement"),
    ("speaker_name", "Speaker"),
    ("speaker_job", "Speaker job"),
    ("speaker_state", "Speaker state"),
    ("speaker_affiliation", "Speaker affiliation"),
    ("subjects", "Subjects"),
    ("statement_context", "Context"),
]

PREDICT_SYSTEM = (
    "You are a careful fact-checking analyst. You assess whether a public "
    "statement made by a politician or public figure is, on balance, TRUTHFUL "
    "or NOT TRUTHFUL, using your prior knowledge of facts and the context "
    "provided. Statements that are largely accurate (even if imperfect) count "
    "as truthful; statements that are misleading, cherry-picked, or false count "
    "as not truthful. Respond ONLY with a compact JSON object of the form "
    '{"verdict": "true" | "false", "confidence": <number between 0 and 1>}. '
    "`confidence` is how sure you are that your verdict is correct."
)

EXPLAIN_SYSTEM = (
    "You are a careful fact-checking analyst. Assess whether a public statement "
    "is, on balance, TRUTHFUL or NOT TRUTHFUL, and explain the specific factors "
    "driving your judgement. Base the explanation only on the actual statement "
    "and metadata provided plus your factual knowledge; do not invent evidence. "
    "Respond ONLY with a compact JSON object of the form "
    '{"verdict": "true" | "false", "confidence": <0..1>, '
    '"key_factors": [<3-5 short phrases naming the concrete signals you used>], '
    '"rationale": "<2-3 sentences explaining how those factors lead to THIS '
    'verdict for THIS statement>"}.'
)

# Single-token target/format for the fine-tuned model. The model is tuned to
# emit exactly "true" or "false", which lets us read a calibrated probability
# straight from the token logprobs at inference time.
FT_SYSTEM = (
    "Classify whether the statement is truthful. Answer with a single word: "
    "true or false."
)


def render_point(point: dict) -> str:
    """Render a point's attributes into a compact, model-friendly block."""
    lines: list[str] = []
    for key, label in _FIELD_ORDER:
        val = point.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        text = str(val).strip()
        if key == "subjects":
            text = ", ".join(p for p in text.split("$") if p)
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def predict_messages(point: dict) -> list[dict]:
    return [
        {"role": "system", "content": PREDICT_SYSTEM},
        {"role": "user", "content": render_point(point)},
    ]


def explain_messages(point: dict) -> list[dict]:
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": render_point(point)},
    ]


def ft_messages(point: dict) -> list[dict]:
    """Inference-time messages for the tuned model. Same system instruction and
    user rendering as the training examples, so train and serve are consistent."""
    return [
        {"role": "system", "content": FT_SYSTEM},
        {"role": "user", "content": render_point(point)},
    ]


def ft_training_record(point: dict, target: str) -> dict:
    """One Vertex AI supervised-tuning JSONL record for a labelled point.

    Format per the Gemini tuning data spec: an optional `systemInstruction`
    plus a `contents` conversation ending in the gold `model` turn.
    """
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": FT_SYSTEM}],
        },
        "contents": [
            {"role": "user", "parts": [{"text": render_point(point)}]},
            {"role": "model", "parts": [{"text": target}]},
        ],
    }
