import torch
from src.data_processing import WikiDataset
from src.model import LanguageModel
from utils.config_manager import config_manager
import warnings

warnings.filterwarnings("ignore")


def generate(prompt, max_seq_len, temperature, model, tokenizer, top_k=50, top_p=0.92, device="cpu", num_pred=3):
    """
    Clean generation with post-processing to remove unwanted tokens
    """
    model.eval()
    preds = []

    for _ in range(num_pred):
        seq = prompt
        indices = tokenizer.encode(prompt, add_special_tokens=False)
        token_counts = {}

        for i in range(max_seq_len):
            src = torch.LongTensor(indices).unsqueeze(1).to(device)

            with torch.no_grad():
                output = model(src)

            logits = output[-1, 0, :].clone()
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = torch.ones_like(logits) * float("-inf")
                mask[top_k_indices] = top_k_values
                logits = mask

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=0)

            # Remove UNK completely
            unk_id = tokenizer.unk_token_id
            if unk_id is not None and unk_id < len(probs):
                probs[unk_id] = 0.0

            # Remove special tokens
            special_tokens = ["[CLS]", "[SEP]", "[MASK]", "<pad>"]
            for special in special_tokens:
                special_id = tokenizer.convert_tokens_to_ids(special)
                if special_id is not None and special_id < len(probs):
                    probs[special_id] = 0.0

            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                continue

            next_token_id = torch.multinomial(probs, 1).item()
            token = tokenizer.decode([next_token_id])

            # Post-process token
            if token in ["<unk>", "[UNK]"]:
                continue
            if token == "@-@":
                token = " "
            elif token == "@":
                continue
            elif token.startswith("##"):
                token = token[2:]
            elif token.startswith("Ġ"):
                token = " " + token[1:]
            elif token.startswith(" ") and len(token) > 1:
                pass
            elif not token.startswith(" ") and token not in [".", ",", "!", "?", ";", ":", "'", '"']:
                token = " " + token

            # Avoid excessive repetition
            if next_token_id in token_counts and token_counts[next_token_id] > 2:
                continue

            token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
            seq += token

            # Stop at sentence end
            if token.strip() in [".", "!", "?"] and len(seq.split()) > 5:
                break

            indices.append(next_token_id)

        # Final cleanup
        seq = seq.replace(" .", ".").replace(" ,", ",").replace("  ", " ")
        seq = seq.replace(" 's", "'s").replace(" n't", "n't")
        preds.append(seq)

    return preds


if __name__ == "__main__":
    architecture = config_manager.get("model", {}).get("architecture", {})
    device = torch.device(config_manager.get("system.device", "cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")

    # Load dataset and tokenizer
    wiki = WikiDataset()

    vocab_size = len(wiki.tokenizer)
    print(f"Vocabulary size: {vocab_size}")

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
    model_path = config_manager.get("paths", {}).get("models", {}).get("saved", "./models/saved")
    best_model_path = f"{model_path}/best_model.pt"

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))

        print(f"✅ Loaded model from {best_model_path}")
    except FileNotFoundError:
        print(f"⚠️ Model not found at {best_model_path}, using random weights")

    # Generate text with different prompts
    prompts = [
        "In a galaxy far, far away, there",
        "The sun was setting in the",
        "Once upon a time, there lived a young princess named",
        "What is the meaning of",
    ]

    for prompt in prompts:
        print("\n" + "=" * 50)
        print(f"Prompt: '{prompt}'")
        print("=" * 50)

        generations = generate(
            prompt=prompt,
            max_seq_len=35,
            temperature=0.65,
            top_k=50,
            top_p=0.92,
            model=model,
            tokenizer=wiki.tokenizer,
            device=device,
            num_pred=3,
        )

        for i, gen in enumerate(generations, 1):
            print(f"Generation {i}: {gen}")
            print()
