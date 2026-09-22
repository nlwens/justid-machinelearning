"""Create the two Hub repos and upload the fine-tuned BERTje folders.

Run after `hf auth login` (Write token):

    .venv\\Scripts\\python.exe upload_models.py
"""

from __future__ import annotations

from huggingface_hub import HfApi, create_repo, whoami

import config

ALLOW = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "labels.json",
    "eval.json",
    "README.md",
]

RG_CARD = """---
language: nl
license: apache-2.0
library_name: transformers
tags:
  - legal
  - dutch
  - multi-label
  - text-classification
base_model: GroNLP/bert-base-dutch-cased
---

# JustID rechtsgebieden

BERTje fine-tuned to assign one or more Dutch law-area labels to a court ruling.
A label is kept if its sigmoid score is at least 0.5.

Code and notebooks: [nlwens/justid-machinelearning](https://github.com/nlwens/justid-machinelearning)
"""

BK_CARD = """---
language: nl
license: apache-2.0
library_name: transformers
tags:
  - legal
  - dutch
  - multi-label
  - text-classification
base_model: GroNLP/bert-base-dutch-cased
---

# JustID bijzondere kenmerken

BERTje fine-tuned to assign one or more bijzondere-kenmerken labels to a court ruling.
A label is kept if its sigmoid score is at least 0.5.

Code and notebooks: [nlwens/justid-machinelearning](https://github.com/nlwens/justid-machinelearning)
"""


def upload_one(local, repo_id: str, card: str) -> None:
    readme = local / "README.md"
    readme.write_text(card, encoding="utf-8")
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=ALLOW,
    )
    print("uploaded", repo_id)


def main() -> None:
    user = whoami()["name"]
    print("logged in as", user)
    if user != "newnus":
        print("expected Hugging Face user newnus")
    upload_one(config.RG_MODEL_DIR, config.RG_HUB_ID, RG_CARD)
    upload_one(config.BK_MODEL_DIR, config.BK_HUB_ID, BK_CARD)
    print("done")


if __name__ == "__main__":
    main()
