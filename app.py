"""
app.py — Cognitive Bias Detection Flask Server
================================================

WHAT'S FIXED IN THIS VERSION
──────────────────────────────
1. Model loads roberta-base fine-tuned bias_model correctly
   Label convention:  0 = Non-Biased,  1 = Biased

2. Inference prepends [OPINION] prefix — model was trained with
   these prefixes so inference must use them too.

3. REWRITE_MAP massively expanded — covers "hellbent", "ginning up",
   "witch hunt", "peddling", "cronies", "regime", "propaganda" etc.

4. rewrite_text() always produces output for biased sentences —
   never returns the original unchanged text.

5. NaN% confidence bug fixed — confidence is a clean formatted string.

HOW TO RUN:
  pip install flask transformers torch
  python app.py
"""

import os
import re
import torch
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Model
# ─────────────────────────────────────────────────────────────────────────────

device = 0 if torch.cuda.is_available() else -1

INFERENCE_PREFIX = "[OPINION] "   # prepended at inference — matches training
LABEL_BIASED     = None
classifier       = None


def _try_load_local():
    if not os.path.isdir("bias_model"):
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained("bias_model")
        mdl = AutoModelForSequenceClassification.from_pretrained("bias_model")
        clf = pipeline(
            "text-classification",
            model=mdl, tokenizer=tok,
            device=device, truncation=True, max_length=256,
        )
        # Determine which label string means Biased from model config
        config_labels = mdl.config.id2label
        label_biased  = next(
            (v for v in config_labels.values()
             if "bias" in v.lower() and "non" not in v.lower()),
            "LABEL_1"
        )
        print(f"Loaded LOCAL bias_model. Biased label = '{label_biased}'")
        return clf, label_biased
    except Exception as e:
        print(f"Could not load local bias_model: {e}")
        return None, None


def _load_public():
    global INFERENCE_PREFIX
    INFERENCE_PREFIX = ""   # public model has no prefix
    print("Loading public model: valurank/distilroberta-bias ...")
    clf = pipeline(
        "text-classification",
        model="valurank/distilroberta-bias",
        device=device, truncation=True, max_length=256,
    )
    print("Public model loaded.")
    return clf, "LABEL_0"


classifier, LABEL_BIASED = _try_load_local()
if classifier is None:
    try:
        classifier, LABEL_BIASED = _load_public()
    except Exception as e:
        print("FATAL: could not load any model:", e)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Word lists
# ─────────────────────────────────────────────────────────────────────────────

BIAS_WORDS = [
    "clearly","obviously","undeniably","indisputably","unquestionably",
    "inevitably","of course","naturally","everyone knows","needless to say",
    "worst","terrible","horrifying","catastrophic","disastrous","disgusting",
    "appalling","outrageous","scandalous","shameful","pathetic","ridiculous",
    "absurd","idiotic","stupid","insane","despicable","alarming","shocking",
    "devastating","radical","extremist","corrupt","fraudulent","hypocritical",
    "divisive","reckless","irresponsible","incompetent","dangerous","evil",
    "hateful","racist","bigoted","hellbent","ginning up","peddling","cronies",
    "witch hunt","deep state","fake","rigged","hoax","mob","thug","elites",
    "globalist","puppet","regime","propaganda","woke","grooming",
    "indoctrination","invasion","slammed","blasted","attacked","destroyed",
    "crushed","exposed","forced","refused","failed","pushed",
    "weaponised","weaponized","exploiting","pandering",
]

REWRITE_MAP = {
    # Certainty markers
    "clearly":"it appears","obviously":"it seems","undeniably":"arguably",
    "indisputably":"arguably","unquestionably":"arguably",
    "inevitably":"possibly","of course":"it could be argued",
    "naturally":"understandably",
    # Strong negative adjectives
    "worst":"less effective","terrible":"concerning",
    "horrifying":"troubling","catastrophic":"serious","disastrous":"concerning",
    "disgusting":"objectionable","appalling":"troubling",
    "outrageous":"controversial","scandalous":"disputed","shameful":"regrettable",
    "pathetic":"inadequate","ridiculous":"debatable","absurd":"questionable",
    "idiotic":"unwise","stupid":"misguided","insane":"unusual",
    "despicable":"unacceptable","alarming":"notable","shocking":"notable",
    "devastating":"significant",
    # Politically loaded adjectives
    "radical":"unconventional","extremist":"hardline",
    "corrupt":"alleged to be acting unethically",
    "fraudulent":"allegedly deceptive","hypocritical":"inconsistent",
    "divisive":"contentious","reckless":"hasty",
    "irresponsible":"ill-considered","incompetent":"underperforming",
    "dangerous":"potentially risky","evil":"harmful","hateful":"harmful",
    "racist":"allegedly discriminatory","bigoted":"intolerant",
    # Colloquial loaded phrases
    "hellbent":"determined","ginning up":"increasing","peddling":"promoting",
    "cronies":"associates","witch hunt":"investigation",
    "deep state":"government bureaucracy","fake":"disputed",
    "rigged":"allegedly unfair","hoax":"disputed claim","mob":"crowd",
    "thug":"individual","elites":"officials","globalist":"internationally focused",
    "puppet":"ally","regime":"government","propaganda":"messaging",
    "woke":"progressive","grooming":"influencing",
    "indoctrination":"instruction","invasion":"influx",
    # Loaded verbs
    "slammed":"criticised","blasted":"criticised","attacked":"challenged",
    "destroyed":"undermined","crushed":"defeated","exposed":"revealed",
    "forced":"required","weaponised":"used","weaponized":"used",
    "exploiting":"using","pandering":"appealing",
}

STRUCTURAL_PATTERNS = [
    (
        r"\b(all|every|none|never|always|no one|everyone|everything|"
        r"nothing|everybody|anybody)\b",
        "Uses absolute language — consider qualifying with 'some', 'often', or 'many'."
    ),
    (
        r"\b(the fact that|it is (a )?fact|it is clear|we all know|"
        r"everyone knows|needless to say|of course|naturally)\b",
        "Presupposes the claim is true — consider stating it as an opinion or citing a source."
    ),
    (
        r"\b(just|merely|only|simply)\b",
        "Minimising language — removing 'just/merely/only/simply' gives a more neutral tone."
    ),
    (
        r"\b(slammed|blasted|attacked|destroyed|crushed|failed|refused|"
        r"forced|pushed|exposed|weaponi[sz]ed|pandering|exploiting)\b",
        "Loaded verb — consider a neutral verb such as 'said', 'stated', or 'declined'."
    ),
    (
        r"\b(hellbent|ginning up|peddling|witch hunt|deep state|hoax|rigged|"
        r"mob|thug|regime|puppet|propaganda|grooming|indoctrination|invasion)\b",
        "Politically loaded phrase — consider more neutral and specific language."
    ),
    (
        r"[!]{2,}",
        "Multiple exclamation marks signal heightened emotion — consider restating calmly."
    ),
    (
        r"\b(no sane|everyone knows|any reasonable|obviously true)\b",
        "Sweeping generalisation — specify who holds this view or cite a source."
    ),
]

BIAS_THRESHOLD = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_bias_score(text: str) -> float:
    if not text.strip() or classifier is None:
        return 0.0
    try:
        result = classifier(INFERENCE_PREFIX + text.strip())[0]
        score  = result["score"]
        return score if result["label"] == LABEL_BIASED else 1.0 - score
    except Exception as e:
        print(f"Inference error: {e}")
        return 0.0


def classify_bias(score: float) -> str:
    if score < 0.20: return "Neutral"
    if score < 0.35: return "Very slightly biased"
    if score < 0.50: return "Slightly biased"
    if score < 0.65: return "Moderately biased"
    if score < 0.80: return "Biased"
    return "Strongly biased"


def split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])(?:\s+|\n+|$)', text.strip())
    return [s.strip() for s in raw if s.strip()]


def highlight_bias_words(text: str) -> str:
    pattern = r"\b(" + "|".join(re.escape(w) for w in BIAS_WORDS) + r")\b"
    return re.sub(
        pattern,
        lambda m: f"<span class='bias'>{m.group(0)}</span>",
        text, flags=re.IGNORECASE,
    )


_REWRITE_PATTERN = r"\b(" + "|".join(re.escape(w) for w in REWRITE_MAP) + r")\b"

def _rewrite_replacer(match):
    original = match.group(0)
    neutral  = REWRITE_MAP.get(original.lower(), original)
    if original.isupper():        return neutral.upper()
    if original[0].isupper():    return neutral.capitalize()
    return neutral


def rewrite_text(text: str, is_biased: bool = True):
    """
    Returns (rewritten_text, rewrite_type).
    NEVER returns the original unchanged text for a biased sentence.
    """
    if not is_biased:
        return text, "none"

    # Pass 1 — keyword substitution
    rewritten = re.sub(_REWRITE_PATTERN, _rewrite_replacer, text, flags=re.IGNORECASE)
    if rewritten.lower() != text.lower():
        return rewritten, "keyword"

    # Pass 2 — structural pattern suggestion
    for pat, suggestion in STRUCTURAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return f"{text}\n\n Suggestion: {suggestion}", "structural"

    # Pass 3 — generic fallback (model detected bias, no specific pattern found)
    return (
        f"{text}\n\n Suggestion: Bias detected in framing or tone. "
        "Consider using neutral verbs ('said', 'stated', 'announced'), "
        "citing specific evidence, and presenting multiple perspectives.",
        "structural",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if classifier is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty input."}), 400

        sentences   = split_sentences(text)
        results     = []
        total_score = 0.0

        for sent in sentences:
            if not sent:
                continue
            score     = get_bias_score(sent)
            is_biased = score >= BIAS_THRESHOLD
            total_score += score
            rewritten_text, rewrite_type = rewrite_text(sent, is_biased)

            results.append({
                "sentence":     sent,
                "score":        round(score, 3),
                "confidence":   f"{score * 100:.1f}%",
                "label":        classify_bias(score),
                "highlighted":  highlight_bias_words(sent),
                "rewritten":    rewritten_text,
                "rewrite_type": rewrite_type,
                "is_biased":    is_biased,
            })

        if not results:
            return jsonify({"error": "No valid sentences found."}), 400

        overall_bias = total_score / len(results)
        most_biased  = max(results, key=lambda r: r["score"])

        full_rewritten = re.sub(
            _REWRITE_PATTERN, _rewrite_replacer, text, flags=re.IGNORECASE
        )
        if full_rewritten.lower() == text.lower():
            full_rewritten = (
                text + "\n\n💡 No direct keyword matches. "
                "See per-sentence suggestions above for structural rewrites."
            )

        return jsonify({
            "overall_bias":         round(overall_bias, 3),
            "overall_label":        classify_bias(overall_bias),
            "overall_confidence":   f"{overall_bias * 100:.1f}%",
            "sentence_count":       len(results),
            "sentences":            results,
            "highlighted":          highlight_bias_words(text),
            "rewritten":            full_rewritten,
            "most_biased_sentence": most_biased["sentence"],
            "most_biased_score":    most_biased["score"],
        })

    except Exception as e:
        print("ERROR in /analyze:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)