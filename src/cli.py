import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.config import DEFAULT_CONFIG, ROOT, load_config
from src.indexing import IndexStore, StaleIndexError
from src.ingestion import default_manifest_path, load_manifest_inputs
from src.search import SEARCH_METHODS, resolve_methods, search


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokalna pretraga PDF/TXT materijala.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index", help="Dodaj dokumente iz manifesta.")
    build.add_argument("--manifest", type=Path, default=default_manifest_path())
    commands.add_parser("rebuild", help="Ponovo izgradi indeks sacuvanih izvora.")
    commands.add_parser("upgrade-bm25", help="Dodaj BM25 bez promene postojecih odlomaka i vektora.")
    query = commands.add_parser("search", help="Pretrazi postojeci indeks.")
    query.add_argument("query")
    query.add_argument("--method", choices=[*SEARCH_METHODS, "both", "all"], default="all",
                       help="all: tri metode; both: prethodni semanticki/TF-IDF par.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        store = IndexStore(args.index_dir)
        if args.command in {"index", "rebuild", "upgrade-bm25"}:
            if args.command == "upgrade-bm25":
                result = store.upgrade_bm25(config)
            else:
                from src.model import SemanticEncoder

                encoder = SemanticEncoder(config)
                if args.command == "index":
                    sources = load_manifest_inputs(args.manifest)
                    result = store.add(sources, encoder)
                else:
                    result = store.rebuild(encoder)
            index = store.load(config)
            payload = {
                "success": True, **asdict(result),
                "document_count": len(index.documents),
                "chunk_count": len(index.chunks),
                "bm25_ready": index.bm25_matrix is not None,
                "document_warnings": [
                    {"document_id": doc.document_id, "file_name": doc.file_name,
                     "warnings": doc.warnings}
                    for doc in index.documents if doc.warnings
                ],
            }
        else:
            index = store.load(config)
            methods = resolve_methods(args.method)
            if "bm25" in methods and index.bm25_matrix is None:
                raise StaleIndexError("Indeks jos nema BM25. Pokrenite upgrade-bm25.")
            encoder = None
            if "semantic" in methods:
                from src.model import SemanticEncoder

                encoder = SemanticEncoder(config)
            payload = {
                "success": True,
                "results": [asdict(search(index, args.query, method, encoder))
                            for method in methods],
            }
    except (ValueError, OSError) as error:
        print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=True))
        return 1
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
