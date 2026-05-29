import torch
from torch import nn
from transformers import AutoTokenizer
from utils.config_manager import config_manager


class WeightDrop(nn.Module):
    """WeightDrop wrapper for RNN modules"""

    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        if issubclass(type(self.module), nn.RNNBase):
            self.module.flatten_parameters = lambda: None

            for name_w in self.weights:
                print(f"Applying weight drop of {self.dropout} to {name_w}")
                w = getattr(self.module, name_w)
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w + "_raw", nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            mask = torch.nn.functional.dropout(torch.ones_like(raw_w), p=self.dropout, training=self.training)
            # Scale the weights
            if self.training:
                mask = mask / (1 - self.dropout)
            setattr(self.module, name_w, raw_w * mask)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """Apply dropout to embedding weights"""
    if dropout and embed.training:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1))
        mask = mask.bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx if embed.padding_idx is not None else -1

    return torch.nn.functional.embedding(
        words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse
    )


class LockedDropout(nn.Module):
    """LockedDropout - same dropout mask for all time steps"""

    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class LanguageModel(nn.Module):
    """AWD-LSTM Language Model"""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropoute=0.2,
        dropouti=0.2,
        dropouth=0.2,
        dropouto=0.2,
        weight_drop=0.2,
        tie_weights=True,
    ):
        super().__init__()

        self.config = config_manager
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("model", {}).get("tokenizer_name", "bert-base-uncased")
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tie_weights = tie_weights

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        # LSTM layers
        self.lstms = nn.ModuleList()

        # First LSTM: embedding_dim -> hidden_dim
        self.lstms.append(nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False))

        # Middle LSTMs: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            self.lstms.append(nn.LSTM(hidden_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False))

        # Last LSTM: hidden_dim -> embedding_dim (if num_layers > 1)
        if num_layers > 1:
            self.lstms.append(nn.LSTM(hidden_dim, embedding_dim, num_layers=1, dropout=0, batch_first=False))

        # Apply WeightDrop if specified
        if weight_drop > 0:
            for i, lstm in enumerate(self.lstms):
                self.lstms[i] = WeightDrop(lstm, ["weight_hh_l0"], dropout=weight_drop)

        # Output layer
        self.fc = nn.Linear(embedding_dim, vocab_size)

        # Tie weights between embedding and output layer
        if tie_weights:
            self.fc.weight = self.embedding.weight

        # Dropout layers
        self.lockdrop = LockedDropout()
        self.dropoute = dropoute  # embedding dropout
        self.dropouti = dropouti  # input dropout
        self.dropouth = dropouth  # hidden dropout
        self.dropouto = dropouto  # output dropout

    def forward(self, src, hidden=None):
        """
        Forward pass of the model
        src: [seq_len, batch_size]
        hidden: tuple of (h0, c0) for each LSTM layer
        """
        # Embedding with dropout
        emb = embedded_dropout(self.embedding, src, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        # Pass through LSTM layers
        new_hidden = []
        for i, lstm in enumerate(self.lstms):
            emb, h = lstm(emb, hidden[i] if hidden is not None else None)
            new_hidden.append(h)

            # Apply dropout between layers (except last layer)
            if i != len(self.lstms) - 1:
                emb = self.lockdrop(emb, self.dropouth)

        # Apply output dropout
        emb = self.lockdrop(emb, self.dropouto)

        # Output layer
        output = self.fc(emb)

        return output

    def init_hidden(self, batch_size, device):
        """Initialize hidden states"""
        hidden = []
        for lstm in self.lstms:
            h0 = torch.zeros(1, batch_size, lstm.hidden_size).to(device)
            c0 = torch.zeros(1, batch_size, lstm.hidden_size).to(device)
            hidden.append((h0, c0))
        return hidden


if __name__ == "__main__":
    # Get vocabulary size from config or dataset
    # Note: You need to define `vocab` before using this
    architecture = config_manager.get("model", {}).get("architecture", {})

    # Example vocab size (replace with actual vocab from your dataset)
    vocab_size = 50000

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
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    seq_len = 70
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_input = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
    hidden = model.init_hidden(batch_size, device)
    output, new_hidden = model(test_input, hidden)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
