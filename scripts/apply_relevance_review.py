import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT, load_config
from src.evaluation import load_queries, validate_queries
from src.indexing import IndexStore


DRAFT = ROOT / "data" / "evaluation" / "queries_draft.json"
OUTPUT = ROOT / "data" / "evaluation" / "queries_v1.json"
AUDIT = ROOT / "data" / "evaluation" / "review_audit_v1.json"
EXPECTED_DRAFT = "9388f0df76e2214abf2cb6a8dc4bc6a8dbbdbe9fe14eb4d8e8e1647bdaaa2c3b"


def main() -> None:
    original = DRAFT.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_DRAFT:
        raise ValueError("Draft changed; the recorded source review must not be applied blindly.")
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("Review artifacts already exist; do not overwrite them.")
    index = IndexStore().load(load_config())
    load_queries(DRAFT, index, allow_draft=True)
    payload = copy.deepcopy(json.loads(original))
    queries = {query["query_id"]: query for query in payload["queries"]}
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    changes: dict[str, list[str]] = {}

    def resolve(prefix: str) -> str:
        matches = [chunk_id for chunk_id in chunks if chunk_id.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Evidence prefix is not unique: {prefix}")
        return matches[0]

    def revise(query_id: str, reason: str, **fields) -> None:
        queries[query_id].update(fields)
        changes.setdefault(query_id, []).append(reason)

    def remove(query_id: str, prefix: str, reason: str) -> None:
        chunk_id = resolve(prefix)
        query = queries[query_id]
        query["relevant_chunk_ids"] = [
            value for value in query["relevant_chunk_ids"] if value != chunk_id
        ]
        query["evidence"] = [
            support for support in query["evidence"] if support["chunk_id"] != chunk_id
        ]
        changes.setdefault(query_id, []).append(reason)

    general_questions = {
        "dev-001": "Why is regularization used when training a model?",
        "dev-002": "What fitting problem does a complexity penalty help guard against?",
        "dev-003": "What can happen when a model fits its training data without a complexity penalty?",
    }
    for query_id, text in general_questions.items():
        revise(query_id, "Generalized the wording to match both generic and logistic-regression evidence.",
               query=text, intent_group="dev-regularization-purpose",
               answer_summary="Regularization reduces the risk of fitting the training data too closely and failing to generalize.")
    for query_id in ("dev-004", "dev-005", "dev-006"):
        remove(query_id, "chunk_0b412e", "Removed an introductory passage whose explanatory comparison is cut at the boundary.")
    for query_id in ("dev-009", "dev-010"):
        revise(query_id, "Qualified the lesson's comparison; linear regression can also be regularized.",
               answer_summary="The lesson emphasizes Log Loss instead of squared loss and regularization to control overfitting. This does not mean linear regression cannot use regularization.")

    remove("test-002", "chunk_947456", "Removed a passage describing MAE robustness without independently explaining MSE outlier sensitivity.")
    revise("test-002", "The question explicitly names both loss metrics, so its type is direct.",
           query_type="direct")
    remove("test-003", "chunk_0f55d3", "Removed the introductory chunk because its list of gradient-descent steps is incomplete.")
    revise("test-008", "Restricted the claim to exact mathematics and finite inputs, not floating-point saturation.",
           query="For finite real-valued inputs in the mathematical sigmoid, are 0 and 1 attained or only approached?",
           answer_summary="For finite real inputs in exact arithmetic, the output lies strictly between 0 and 1. The endpoints are approached as limits.")
    revise("test-009", "Made the question explicitly source-specific rather than presenting the lesson's precision explanation as the only general reason.",
           query="What numerical-precision concern does the lesson give for squared loss near sigmoid outputs of 0 and 1?")
    revise("test-010", "Excluded equality at the threshold, which the source says is implementation-dependent.",
           query="For scores strictly above or below a binary classification threshold, which class is assigned?",
           answer_summary="A score above the threshold is assigned to the positive class; a score below it to the negative class. Equality is outside this question and depends on the implementation.")
    revise("test-014", "Specified binary-classification recall and used its standard sensitivity synonym.",
           query="In binary classification, what share of cases does sensitivity, also called recall, measure?")
    remove("test-017", "chunk_8bd9d9", "Removed the introduction: it names ROC but does not explain its TPR/FPR construction.")
    revise("test-020", "Broadened the answer representation to include the independently supported MIT column-vector example.",
           query="In these learning examples, what numerical representation is used for one data instance?",
           answer_summary="A numeric feature vector, represented as an array or column vector of the example's feature values.")
    extra = resolve("chunk_07f151")
    queries["test-020"]["relevant_chunk_ids"].append(extra)
    queries["test-020"]["evidence"].append({
        "chunk_id": extra,
        "quote": "We assume that each image (grayscale) is represented as a column vector x of dimension d. So, the pixel intensity values in the image, column by column, are concatenated into a single column vector.",
    })
    revise("test-022", "Scoped the scaling claim to the lesson's training example, not all possible model families.",
           query="In the lesson's training example, why should numerical features with very different ranges be scaled?")
    revise("test-023", "Scoped the importance claim to the described learner rather than a universal property of all models.",
           query="Why might the learner in the example initially treat a wider-range input as more important?")
    remove("test-027", "chunk_c7878f", "Removed a chunk that gives vector length but not the full one-hot value pattern.")
    remove("test-027", "chunk_66a28d", "Removed an example/pattern chunk without the general N-category length rule.")
    remove("test-028", "chunk_c7878f", "Removed length-only evidence that does not identify the active category marker.")
    revise("test-032", "Preserved the source's rough-rule qualification rather than treating it as a universal sample-size law.",
           query_type="direct",
           answer_summary="The lesson's rough guideline is at least one or two orders of magnitude more examples than trainable parameters; it is not a universal guarantee.")
    remove("test-036", "chunk_12c02d", "Removed a passage whose real-world representativeness criterion is truncated.")
    remove("test-036", "chunk_8e03a7", "Kept the complete passage explicitly relating all four distributions instead of an incomplete cross-chunk list.")
    weights = resolve("chunk_d6619e")
    revise("test-037", "Replaced a question too close to the development regularization-purpose need with a distinct, fully supported weight-distribution question.",
           query="What distribution shape and mean does the lesson associate with model weights under a high regularization rate?",
           intent_group="test-l2-weight-distribution",
           answer_summary="The lesson describes a tendency toward a normal distribution centered at a mean weight of zero, not an unconditional guarantee for every model.",
           relevant_chunk_ids=[weights],
           evidence=[{
               "chunk_id": weights,
               "quote": "A high regularization rate: Strengthens the influence of regularization, thereby reducing the chances of overfitting. Tends to produce a histogram of model weights having the following characteristics: a normal distribution a mean weight of 0.",
           }])
    for query_id in ("test-039", "test-040"):
        revise(query_id, "Distinguished a stopping condition from a guarantee of convergence on nonseparable data.",
               answer_summary="Updates cease when all training examples are classified correctly. This is the stopping condition, not a guarantee that every training set reaches it.")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    for query in payload["queries"]:
        query["review_status"] = "ai_reviewed"
        query["review_notes"] = " ".join(changes.get(query["query_id"], [
            "Source-grounded question, answer and evidence reviewed; no correction required."
        ]))
    payload["annotation_status"] = "ai_reviewed"
    payload["annotation_method"] = (
        "AI-assisted draft followed by independent source-content review and corrections "
        "by the assistant at the user's request. No human/expert review. No test retrieval "
        "rankings were consulted before this review was frozen."
    )
    payload["human_review_complete"] = False
    payload["reviewed_by"] = "assistant"
    payload["reviewed_at_utc"] = reviewed_at
    payload["reviewed_from_sha256"] = EXPECTED_DRAFT
    checked = validate_queries(payload, index, allow_ai_reviewed=True)
    if Counter(query.split for query in checked) != {"dev": 10, "test": 40, "no_answer": 5}:
        raise ValueError("Review changed the agreed split.")
    encoded = (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    audit = {
        "reviewed_at_utc": reviewed_at,
        "reviewer": "assistant",
        "human_review_complete": False,
        "criterion": "A positive chunk must independently supply enough context to answer the question; topic-only or truncated partial answers are insufficient.",
        "queries_reviewed": len(checked),
        "changed_query_ids": sorted(changes),
        "changes": changes,
        "source_draft_sha256": EXPECTED_DRAFT,
        "reviewed_dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "test_rankings_consulted": False,
        "retrieval_configuration_changed": False,
        "limitations": [
            "AI-assisted relevance judgments remain subjective and are not a substitute for independent expert assessment.",
            "Additional relevant passages may still be missed despite source review.",
            "The small corpus, repeated information needs and known PDF glyph errors limit generalization.",
        ],
    }
    with OUTPUT.open("xb") as stream:
        stream.write(encoded)
    with AUDIT.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({
        "queries": len(checked), "changed_queries": len(changes),
        "dataset_sha256": audit["reviewed_dataset_sha256"],
        "annotation_status": "ai_reviewed", "human_review_complete": False,
    }))


if __name__ == "__main__":
    main()
