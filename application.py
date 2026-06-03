from flask import Flask, render_template, request, jsonify
import torch
from src.model import LanguageModel
from src.data_processing import WikiDataset
from utils.config_manager import config_manager
import os

from utils.google_drive_downloader import GoogleDriveDownloader

app = Flask(__name__, template_folder="templates", static_folder="static")

# Punctuation set for cleaning
PUNCTUATION = {
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    '"',
    "'",
    "`",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "/",
    "\\",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "-",
    "_",
    "=",
    "+",
    "~",
}


def clean_token(token):
    """Clean token from prefixes"""
    if token.startswith("##"):
        return token[2:]
    elif token.startswith("Ġ"):
        return token[1:]
    return token


def is_valid_token(token):
    """Check if token is valid (not punctuation, not special)"""
    token_clean = token.strip()
    if token_clean in PUNCTUATION:
        return False
    if len(token_clean) == 1 and not token_clean.isalpha():
        return False
    if token_clean in ["<unk>", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<pad>", "``", "''"]:
        return False
    if not token_clean:
        return False
    return True


# Global loading (once at startup)
device = torch.device(config_manager.get("system.device", "cuda" if torch.cuda.is_available() else "cpu"))
wiki = WikiDataset()
vocab_size = len(wiki.tokenizer)

architecture = config_manager.get("model", {}).get("architecture", {})

# Initialize model
model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=architecture.get("embedding_dim", 300),
    hidden_dim=architecture.get("hidden_dim", 1150),
    num_layers=architecture.get("num_layers", 3),
    dropoute=architecture.get("dropoute", 0.1),
    dropouti=architecture.get("dropouti", 0.65),
    dropouth=architecture.get("dropouth", 0.3),
    dropouto=architecture.get("dropouto", 0.4),
    weight_drop=architecture.get("weight_drop", 0.2),
    tie_weights=True,
).to(device)

# Load best model weights
model_path = config_manager.get("paths", {}).get("models", {}).get("saved", "./artifacts")
best_model_path = f"{model_path}/autocomplete.pt"

try:
    model_file = GoogleDriveDownloader.get_model_path(best_model_path)

    state_dict = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Model loaded successfully from {model_file}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("⚠️ Running without model - predictions will not work")


def predict_next_words(text, temperature=0.65, top_k=8):
    """Predict next three-word suggestions (trigrams) with cleaning. Fallback to single tokens if no trigram found."""

    if not text or len(text.strip()) == 0:
        return []

    input_ids = wiki.tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) == 0:
        return []

    src = torch.LongTensor(input_ids).unsqueeze(1).to(device)
    suggestions = []

    with torch.no_grad():
        output = model(src)
        last_logits = output[-1, 0, :] / temperature
        probs_first = torch.softmax(last_logits, dim=-1)
        top_k_vals, top_k_idxs = torch.topk(probs_first, min(top_k, vocab_size))

        # Try to build trigrams using top-2 for second and third tokens
        for i in range(len(top_k_idxs)):
            first_id = top_k_idxs[i].item()
            first_prob = top_k_vals[i].item()
            raw_first = wiki.tokenizer.decode([first_id])
            first_clean = clean_token(raw_first)
            if not is_valid_token(first_clean):
                continue

            # Second token: take top-2
            extended_ids = input_ids + [first_id]
            extended_src = torch.LongTensor(extended_ids).unsqueeze(1).to(device)
            output2 = model(extended_src)
            last_logits2 = output2[-1, 0, :] / temperature
            probs2 = torch.softmax(last_logits2, dim=-1)
            top2_second_vals, top2_second_idxs = torch.topk(probs2, min(2, vocab_size))

            for j in range(top2_second_idxs.size(0)):
                second_id = top2_second_idxs[j].item()
                second_prob = top2_second_vals[j].item()
                raw_second = wiki.tokenizer.decode([second_id])
                second_clean = clean_token(raw_second)
                if not is_valid_token(second_clean):
                    continue

                # Third token: take top-2
                extended_ids2 = extended_ids + [second_id]
                extended_src2 = torch.LongTensor(extended_ids2).unsqueeze(1).to(device)
                output3 = model(extended_src2)
                last_logits3 = output3[-1, 0, :] / temperature
                probs3 = torch.softmax(last_logits3, dim=-1)
                top2_third_vals, top2_third_idxs = torch.topk(probs3, min(2, vocab_size))

                for k in range(top2_third_idxs.size(0)):
                    third_id = top2_third_idxs[k].item()
                    third_prob = top2_third_vals[k].item()
                    raw_third = wiki.tokenizer.decode([third_id])
                    third_clean = clean_token(raw_third)
                    if not is_valid_token(third_clean):
                        continue

                    joint_prob = first_prob * second_prob * third_prob
                    full_phrase = f"{first_clean} {second_clean} {third_clean}"
                    suggestions.append({"token": full_phrase, "probability": joint_prob})
                    if len(suggestions) >= top_k * 2:  # limit
                        break
                if len(suggestions) >= top_k * 2:
                    break
            if len(suggestions) >= top_k * 2:
                break

    # Sort and keep top_k
    suggestions.sort(key=lambda x: x["probability"], reverse=True)
    trigrams = suggestions[:top_k]

    # Fallback: if no trigram found, return single valid tokens
    if not trigrams:
        for i in range(len(top_k_idxs)):
            first_id = top_k_idxs[i].item()
            first_prob = top_k_vals[i].item()
            raw_first = wiki.tokenizer.decode([first_id])
            first_clean = clean_token(raw_first)
            if is_valid_token(first_clean):
                trigrams.append({"token": first_clean, "probability": first_prob})
            if len(trigrams) >= top_k:
                break
        trigrams.sort(key=lambda x: x["probability"], reverse=True)

    return trigrams


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        temperature = data.get("temperature", 0.65)
        top_k = data.get("top_k", 8)

        if not prompt:
            return jsonify({"suggestions": []})

        suggestions = predict_next_words(prompt, temperature, top_k)

        return jsonify({"suggestions": suggestions})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "suggestions": []}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
