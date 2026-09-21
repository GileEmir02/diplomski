import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "search.json"


@dataclass(frozen=True)
class SearchConfig:
    schema_version: int
    model_name: str
    model_revision: str
    device: str
    chunk_tokens: int
    overlap_tokens: int
    top_k: int

    def __post_init__(self) -> None:
        for name in ("schema_version", "chunk_tokens", "overlap_tokens", "top_k"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"Podesavanje {name} mora biti ceo broj.")
        if self.schema_version != 1:
            raise ValueError("Nepodrzana verzija konfiguracije.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Ime modela nije zadato.")
        if not isinstance(self.model_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.model_revision
        ):
            raise ValueError("Revizija modela mora biti tacan commit SHA.")
        if self.device != "cpu":
            raise ValueError("Osnovna verzija podrzava samo CPU.")
        if self.chunk_tokens <= 0 or not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Preklapanje mora biti nenegativno i manje od odlomka.")
        if self.top_k != 5:
            raise ValueError("Osnovni protokol zahteva top_k=5.")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> SearchConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = set(SearchConfig.__dataclass_fields__)
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Konfiguracija mora sadrzati tacno dokumentovana polja.")
    return SearchConfig(**payload)
