# Cell 2: Mục 13 - Classification Head
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


LAB_DIR = Path(__file__).resolve().parent
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
LABELS = {0: "negative", 1: "positive"}


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


class SentimentClassifier(nn.Module):
    def __init__(self, *, pretrained=True, frozen=False):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, add_pooling_layer=False
            )
        else:
            config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.frozen = frozen

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = hidden.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_embedding))


def inspect_classifier():
    torch.manual_seed(42)
    model = SentimentClassifier().eval()
    tokenizer = load_tokenizer()
    encoded = tokenizer(
        ["The acting was excellent.", "The plot was disappointing."],
        padding=True,
        truncation=True,
        max_length=128,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded)
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "head": str(model.classifier),
        "batch_size": len(encoded["input_ids"]),
        "logits_shape": list(logits.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    path = LAB_DIR / "outputs" / "classification_head.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = json.dumps(report, indent=2)
    path.write_text(formatted + "\n", encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    inspect_classifier()
