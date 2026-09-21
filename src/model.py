import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.config import ROOT, SearchConfig, load_config
from src.chunking import count_tokens


MODEL_CACHE = ROOT / ".cache" / "sentence-transformers"
MODEL_REPORT = ROOT / "artifacts" / "preparation" / "model.json"


class SemanticEncoder:
    def __init__(self, config: SearchConfig, *, local_files_only: bool = True) -> None:
        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            revision=config.model_revision,
            device=config.device,
            cache_folder=str(MODEL_CACHE),
            local_files_only=local_files_only,
            trust_remote_code=False,
            token=False,
        )
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length
        dimension = self.model.get_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Model nije prijavio ispravnu dimenziju vektora.")
        if not isinstance(self.max_seq_length, int) or self.max_seq_length <= 0:
            raise ValueError("Model nije prijavio ispravan token limit.")
        if not self.tokenizer.is_fast:
            raise ValueError("Za ocuvanje raspona teksta potreban je fast tokenizer.")
        self.dimension = dimension
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        if config.chunk_tokens + self.special_tokens > self.max_seq_length:
            raise ValueError("Odlomak sa specijalnim tokenima ne staje u model.")

    def token_count(self, text: str) -> int:
        return count_tokens(self.tokenizer, text, add_special_tokens=True)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if isinstance(texts, str) or not texts:
            raise ValueError("Nema tekstova za kodiranje.")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Tekst na poziciji {index} je prazan ili neispravan.")
            count = self.token_count(text)
            if count > self.max_seq_length:
                raise ValueError(
                    f"Tekst na poziciji {index} ima {count} tokena; "
                    f"limit je {self.max_seq_length}. Tekst nije odsecen."
                )
        vectors = np.asarray(
            self.model.encode(
                list(texts),
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError("Model je vratio neocekivan oblik matrice.")
        if not np.isfinite(vectors).all():
            raise ValueError("Model je vratio nekonacne vrednosti.")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0) or not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Model nije vratio normalizovane nenulte vektore.")
        return vectors


def prepare_model(*, allow_download: bool = False, report_path: Path = MODEL_REPORT) -> None:
    config = load_config()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "revision": config.model_revision,
        "config_fingerprint": config.fingerprint(),
        "device": config.device,
        "license_verified": False,
        "license_reference": "config/model_provenance.json",
        "ready": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    encoder = SemanticEncoder(config, local_files_only=not allow_download)
    probe = [
        "A model can overfit its training examples.",
        "Regularization can help reduce overfitting.",
    ]
    first = encoder.encode(probe)
    offline = SemanticEncoder(config, local_files_only=True)
    second = offline.encode(probe)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
    report.update({
        "max_seq_length": encoder.max_seq_length,
        "special_tokens": encoder.special_tokens,
        "embedding_dimension": encoder.dimension,
        "probe_shape": list(first.shape),
        "probe_norms": np.linalg.norm(first, axis=1).tolist(),
        "local_reload_verified": True,
        "ready": True,
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priprema lokalnog semantickog modela.")
    parser.add_argument("--download", action="store_true",
                        help="Dozvoli preuzimanje javnih modelskih fajlova.")
    arguments = parser.parse_args()
    prepare_model(allow_download=arguments.download)
