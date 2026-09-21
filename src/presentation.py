"""Presentation-only transformations; indexed text and retrieval scores stay untouched."""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from src.bm25 import BM25_PARAMETERS
from src.config import ROOT
from src.evaluation import CODE_FILES, corpus_fingerprint
from src.indexing import SearchIndex
from src.provenance import code_provenance_profile
from src.search import SEARCH_METHODS, SearchMethod


_NUMBER = re.compile(r"[+\-\u2212]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+\-\u2212]?\d+)?")
_PLOT_MARKER = re.compile(r"[a-zA-Z]\)")
_BULLET = re.compile(r"(?:[-*\u2022]\s+|\d+[.)]\s+)")
_PROSE_START = re.compile(
    r"^(?:the|this|these|those|it|we|in|for|a|an|to|if|when|as|with|by|our|there|they|however|note)\s+[^\W\d_]{2,}",
    re.IGNORECASE,
)
_MATH_SIGN = re.compile(r"[=+*/^_\u2212\u00d7\u2217\u2211\u220f\u2202\u2207\u2264\u2265\u2260\u02c6\u0370-\u03ff]")
_FORMULA_OPERATOR = re.compile(r"[=+*/^\u2212\u00d7\u2217\u2211\u220f\u2264\u2265\u2260]")
_MATH_WORDS = {"log", "exp", "sin", "cos", "tan", "max", "min", "arg", "lim",
               "tr", "cov", "bias", "loss", "mse", "mae", "lambda"}
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
})


@dataclass(frozen=True)
class ReadingBlock:
    kind: str
    text: str


@dataclass(frozen=True)
class ReadingView:
    original: str
    paragraphs: tuple[str, ...]
    numeric_fragments: tuple[str, ...]
    folded_lines: int
    replacement_characters: int
    formula_fragments: tuple[str, ...] = ()
    blocks: tuple[ReadingBlock, ...] = ()


@dataclass(frozen=True)
class ExperimentRow:
    method: SearchMethod
    hit_at_5: float
    mrr_at_5: float
    precision_at_5: float | None = None


@dataclass(frozen=True)
class SavedExperiment:
    rows: tuple[ExperimentRow, ...]
    query_count: int
    intent_count: int
    dataset_sha256: str
    measured_on: str
    evaluator_profile: str = "current"
    human_review_complete: bool = False


def _numeric_line(line: str) -> int | None:
    tokens = line.strip().split()
    if not tokens:
        return None
    numbers = 0
    for token in tokens:
        if _NUMBER.fullmatch(token):
            numbers += 1
        elif not _PLOT_MARKER.fullmatch(token):
            return None
    return numbers


def _formula_line(line: str) -> bool:
    line = line.strip().translate(_LIGATURES)
    if not line or _PROSE_START.match(line) or _BULLET.match(line):
        return False
    if re.fullmatch(r"\(\d+\)", line) or re.fullmatch(r"[A-Za-z\u0370-\u03ff][\u02c6\u2217\u2080-\u2089]*", line):
        return True
    if re.fullmatch(r"[\s\ufffd]+", line):
        return True
    words = re.findall(r"[^\W\d_]+", line)
    prose_words = [word for word in words if len(word) >= 3 and word.lower() not in _MATH_WORDS]
    if len(prose_words) > 2:
        return False
    if _MATH_SIGN.search(line):
        return True
    return bool(words and all(word.lower() in _MATH_WORDS for word in words) and "\ufffd" in line)


def _incomplete_formula(text: str) -> bool:
    return any(text.count(left) != text.count(right) for left, right in (("(", ")"), ("[", "]"), ("{", "}")))


def _prose_paragraphs(lines: list[str]) -> list[str]:
    paragraphs = []
    current = []

    def flush() -> None:
        if current:
            text = "\n".join(current).translate(_LIGATURES)
            text = re.sub(r"\u00ad[ \t]*\n[ \t]*", "", text)
            text = text.replace("\u00ad", "")
            paragraphs.append(re.sub(r"\s+", " ", text).strip())
            current.clear()

    for line in lines:
        if not line.strip():
            flush()
        else:
            if _BULLET.match(line.strip()):
                flush()
            current.append(line.strip())
    flush()
    return paragraphs


def reading_view(text: str) -> ReadingView:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Odlomak za prikaz mora biti neprazan tekst.")
    lines = text.splitlines()
    paragraphs, fragments, formulas, prose, blocks = [], [], [], [], []
    folded_lines = 0
    position = 0

    def flush_prose() -> None:
        values = _prose_paragraphs(prose)
        paragraphs.extend(values)
        blocks.extend(ReadingBlock("prose", value) for value in values)
        prose.clear()

    while position < len(lines):
        if _numeric_line(lines[position]) is not None or _formula_line(lines[position]):
            end = position
            numeric_lines = 0
            formula_lines = 0
            while end < len(lines):
                count = _numeric_line(lines[end])
                formula = count is None and _formula_line(lines[end])
                if count is None and not formula and lines[end].strip():
                    break
                numeric_lines += int(count is not None and count > 0)
                formula_lines += int(formula)
                end += 1
            fragment = "\n".join(lines[position:end]).strip()
            dense_or_damaged = (
                numeric_lines + formula_lines >= 3
                or "\ufffd" in fragment
                or _incomplete_formula(fragment)
            )
            if (formula_lines and dense_or_damaged) or (not formula_lines and numeric_lines >= 3):
                flush_prose()
                (formulas if formula_lines else fragments).append(fragment)
                folded_lines += sum(bool(line.strip()) for line in lines[position:end])
                position = end
                continue
            if formula_lines and _FORMULA_OPERATOR.search(fragment):
                flush_prose()
                blocks.append(ReadingBlock("formula", fragment))
                position = end
                continue
        prose.append(lines[position])
        position += 1
    flush_prose()
    return ReadingView(text, tuple(paragraphs), tuple(fragments), folded_lines,
                       text.count("\ufffd"), tuple(formulas), tuple(blocks))


def reading_html(view: ReadingView) -> str:
    return '<div class="reading-passage">' + "".join(
        f'<pre class="readable-formula">{escape(block.text)}</pre>'
        if block.kind == "formula" else f"<p>{escape(block.text)}</p>"
        for block in view.blocks
    ) + "</div>"


def markdown_label(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>~-])", r"\\\1", text)


def _quality(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Sacuvana metrika nije konacan broj izmedju nule i jedinice.")
    return float(value)


def load_saved_experiment(path: Path, index: SearchIndex) -> SavedExperiment:
    if index.bm25_vectorizer is None or index.bm25_matrix is None:
        raise ValueError("Za pore\u0111enje sve tri metode prvo pripremite BM25 za aktivni indeks.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Neispravan sacuvani eksperiment.")
    expanded = payload.get("protocol_id", "legacy-v1") != "legacy-v1"
    annotation = payload.get("annotation_mode")
    human_review = payload.get("human_review_complete")
    if (not isinstance(annotation, str)
            or annotation not in ({"reviewed", "ai_source_reviewed"} if expanded else {"ai_source_reviewed"})
            or human_review is not (annotation == "reviewed")
            or payload.get("metrics_recomputed_from_raw_rankings") is not True):
        raise ValueError("Nedostaje potvrda zavrsenog eksperimenta i porekla AI oznaka.")
    count = payload.get("test_queries")
    intents = payload.get("distinct_test_intent_groups")
    if (type(count) is not int or type(intents) is not int or not 1 <= intents <= count
            or (not expanded and (count != 40 or intents != 35))):
        raise ValueError("Sacuvani eksperiment nema ocekivani test skup.")
    precision_available = payload.get("precision_label_coverage") == "pooled_top5"
    if precision_available and not expanded:
        raise ValueError("Objavljeni Precision@5 zahteva prosireni protokol i pregled kandidata.")
    if expanded:
        protocol = payload.get("protocol")
        if (not isinstance(protocol, dict)
                or type(payload.get("metrics_schema_version")) is not int
                or payload["metrics_schema_version"] != 2
                or protocol.get("protocol_id") != payload.get("protocol_id")
                or not isinstance(protocol.get("expected_splits"), dict)
                or protocol["expected_splits"].get("test") != count
                or protocol.get("k") != 5 or protocol.get("precision_denominator") != 5
                or not precision_available
                or not isinstance(payload.get("protocol_sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", payload["protocol_sha256"])):
            raise ValueError("Prosireni rezultat nema potvrden protokol i pregled kandidata za Precision@5.")
    dataset_sha = payload.get("dataset_sha256")
    if not isinstance(dataset_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", dataset_sha):
        raise ValueError("Nedostaje identitet evaluacionog skupa.")
    summaries = payload.get("method_summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(SEARCH_METHODS):
        raise ValueError("Nisu sacuvani zavrsni rezultati sve tri metode.")
    per_type = payload.get("quality_by_query_type")
    if not isinstance(per_type, dict) or set(per_type) != set(SEARCH_METHODS):
        raise ValueError("Nisu sacuvani rezultati po tipu upita za sve tri metode.")
    expected_code = {
        str(Path("src") / name): hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
        for name in CODE_FILES
    }
    fingerprint = corpus_fingerprint(index)
    rows = []
    dates = []
    code_profiles = set()
    type_counts = None
    for method in SEARCH_METHODS:
        row = summaries[method]
        if (not isinstance(row, dict) or row.get("status") != "completed"
                or row.get("method") != method or row.get("split") != "test"
                or row.get("query_count") != count
                or row.get("annotation_mode") != annotation
                or row.get("human_review_complete") is not human_review
                or row.get("dataset_sha256") != dataset_sha):
            raise ValueError("Eksperiment jos nije potpun ili rezultati nisu uskladjeni.")
        profile = code_provenance_profile(row.get("code_sha256"), expected_code, dataset_sha)
        if (row.get("generation") != index.generation
                or row.get("corpus_fingerprint") != fingerprint
                or row.get("config") != asdict(index.config)
                or profile is None
                or row.get("bm25_parameters") != BM25_PARAMETERS):
            raise ValueError(
                "Sacuvana merenja pripadaju drugoj verziji kolekcije, koda ili podesavanja. "
                "Ne prikazuju se kao rezultati trenutnog indeksa."
            )
        code_profiles.add(profile)
        if expanded and any(row.get(key) != payload.get(key)
                            for key in ("protocol_id", "protocol_sha256", "precision_label_coverage",
                                        "metrics_schema_version")):
            raise ValueError("Metode nemaju isti prosireni protokol i pregled kandidata.")
        hit, mrr = _quality(row.get("hit_at_5")), _quality(row.get("mrr_at_5"))
        if mrr > hit:
            raise ValueError("Sacuvani MRR@5 ne moze biti veci od Hit@5.")
        precision = _quality(row.get("precision_at_5")) if precision_available else None
        if precision is not None and precision > hit:
            raise ValueError("Sacuvani Precision@5 ne moze biti veci od Hit@5.")
        typed_rows = per_type[method]
        if not isinstance(typed_rows, dict) or set(typed_rows) != {"direct", "paraphrase", "synonym"}:
            raise ValueError("Eksperiment nema sve tipove test upita.")
        counts = {}
        for kind, typed in typed_rows.items():
            if (not isinstance(typed, dict) or type(typed.get("query_count")) is not int
                    or typed["query_count"] <= 0):
                raise ValueError("Broj upita po tipu nije ispravan.")
            counts[kind] = typed["query_count"]
            if _quality(typed.get("mrr_at_5")) > _quality(typed.get("hit_at_5")):
                raise ValueError("Sacuvani MRR@5 ne moze biti veci od Hit@5.")
            if precision_available and _quality(typed.get("precision_at_5")) > _quality(typed.get("hit_at_5")):
                raise ValueError("Sacuvani Precision@5 ne moze biti veci od Hit@5.")
        if sum(counts.values()) != count or (type_counts is not None and counts != type_counts):
            raise ValueError("Metode nemaju isti potpuni test skup po tipu upita.")
        type_counts = counts
        finished = row.get("finished_at_utc")
        if not isinstance(finished, str):
            raise ValueError("Nedostaje vreme zavrsetka eksperimenta.")
        timestamp = datetime.fromisoformat(finished)
        if timestamp.tzinfo is None:
            raise ValueError("Vreme eksperimenta mora imati navedenu vremensku zonu.")
        dates.append(timestamp)
        rows.append(ExperimentRow(method, hit, mrr, precision))
    if len(code_profiles) != 1:
        raise ValueError("Metode nemaju istu verziju evaluatora.")
    return SavedExperiment(tuple(rows), count, intents, dataset_sha,
                           max(dates).strftime("%d.%m.%Y."), code_profiles.pop(), human_review)
