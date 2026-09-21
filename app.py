import hashlib
from html import escape
import os
from pathlib import Path
from time import perf_counter_ns
from urllib.parse import urlsplit

import streamlit as st

from src.bm25 import BM25_PARAMETERS
from src.config import ROOT, SearchConfig, load_config
from src.indexing import DocumentInfo, IndexStore, SearchIndex
from src.ingestion import (
    BatchImportError, DocumentInput, default_manifest_path, ingest_batch,
    load_manifest_inputs,
)
from src.presentation import (
    load_saved_experiment, markdown_label, reading_html, reading_view,
)
from src.search import SEARCH_METHODS, SearchHit, SearchResponse, search


METHODS = {
    "semantic": {"name": "Semanti\u010dka pretraga", "role": "ZNA\u010cENJE", "color": "#75d8e6",
                 "description": "Povezuje pitanje sa zna\u010denjem odlomaka."},
    "tfidf": {"name": "TF-IDF", "role": "TERMINI", "color": "#d9b575",
              "description": "Pronalazi odlomke pomo\u0107u zajedni\u010dkih termina."},
    "bm25": {"name": "BM25", "role": "TERMINI / DU\u017dINA", "color": "#b9a7e8",
             "description": "Rangira zajedni\u010dke termine uz zasi\u0107enje frekvencije i normalizaciju du\u017eine odlomka."},
}
EXAMPLES = (
    "What is regularization?",
    "How does gradient descent work?",
    "Why use separate test data?",
)
EXPERIMENT = ROOT / os.environ.get(
    "SEMANTIC_SEARCH_REPORT",
    str(Path("artifacts") / "experiments" / "test_three_methods_v2" / "analysis.json"),
)

st.set_page_config(page_title="Semanticka pretraga materijala", layout="wide",
                   initial_sidebar_state="expanded")
st.html(ROOT / "assets" / "app.css")


@st.cache_resource(show_spinner=False)
def get_store(directory: str) -> IndexStore:
    return IndexStore(Path(directory))


@st.cache_resource(show_spinner=False, max_entries=1)
def get_encoder(config: SearchConfig):
    from src.model import SemanticEncoder

    return SemanticEncoder(config)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_index(directory: str, active_key: str, config: SearchConfig) -> SearchIndex:
    return get_store(directory).load(config)


@st.cache_data(show_spinner=False, max_entries=2)
def get_downloads(directory: str, active_key: str) -> dict[str, bytes]:
    return {
        hashlib.sha256(source.content).hexdigest(): source.content
        for source in get_store(directory).source_inputs()
    }


def hint(text: str) -> None:
    st.html(f'<p class="tiny-note">{escape(text)}</p>')


def show_error(error: Exception, message: str) -> None:
    st.error(message)
    if isinstance(error, BatchImportError):
        for issue in error.issues:
            st.text(f"{issue.file_name}: {issue.message}")
    else:
        st.text(str(error))


def clear_results() -> None:
    st.session_state.pop("responses", None)
    st.session_state.pop("search_timings", None)


def clear_search() -> None:
    st.session_state["query_input"] = ""
    clear_results()


def use_example(query: str) -> None:
    st.session_state["query_input"] = query


try:
    config = load_config()
    index_directory = str(Path(os.environ.get(
        "SEMANTIC_SEARCH_INDEX_DIR", str(ROOT / "artifacts" / "indexes")
    )).resolve())
    store = get_store(index_directory)
except (ValueError, OSError) as error:
    show_error(error, "Konfiguracija ili direktorijum indeksa nisu dostupni.")
    st.stop()

index = None
index_error = None
active_key = ""
try:
    if store.active.exists():
        active_key = hashlib.sha256(store.active.read_bytes()).hexdigest()
        index = get_index(index_directory, active_key, config)
except (ValueError, OSError) as error:
    index_error = error

bm25_available = (
    index is not None and index.bm25_vectorizer is not None and index.bm25_matrix is not None
)
available_methods = SEARCH_METHODS if bm25_available else ("semantic", "tfidf")
saved = st.session_state.get("responses", ())
collection_changed = bool(saved) and (
    index is None or any(response.generation != index.generation for response in saved)
)
if collection_changed:
    clear_results()

state_class = "error" if index_error else "" if index else "waiting"
state_label = "INDEKS NIJE SPREMAN" if index_error else "INDEKS SPREMAN" if index else "NEMA KOLEKCIJE"
document_count = len(index.documents) if index else 0
chunk_count = len(index.chunks) if index else 0
method_intro = (
    "Dve spremne metode; BM25 \u010deka pripremu."
    if index is not None and not bm25_available else "Tri metode. Isti izvori."
)
st.html(
    '<header class="desk-header"><div><div class="eyebrow">LOKALNI SISTEM / PRETRAGA IZVORA</div>'
    '<h1>Pretraga materijala</h1>'
    f'<p class="desk-subtitle">{method_intro} Engleski upiti, bez generisanja odgovora.</p>'
    f'</div><div class="header-right"><span class="status-pill {state_class}">{state_label}</span>'
    f'<div class="collection-stats">{document_count:02d} DOKUMENATA &nbsp;/&nbsp; '
    f'{chunk_count} ODLOMAKA &nbsp;/&nbsp; CPU</div></div></header>'
)
prepare_requested = False
app_notices = st.container(key="app_notices")
with app_notices:
    if index_error:
        show_error(index_error, "Aktivni indeks nije spreman. Obnovite ga iz sa\u010duvanih izvora.")
    notice = st.session_state.pop("notice", None)
    if notice:
        st.success(notice)
    for name in st.session_state.pop("duplicates", ()):
        st.text(f"Ve\u0107 postoji, nije duplirano: {name}")
    if collection_changed:
        st.info("Kolekcija je promenjena. Ponovite pretragu za novu verziju.")
    if index is not None and not bm25_available:
        st.info(
            "BM25 jo\u0161 nije pripremljen za ovu kolekciju. Pore\u0111enje do pripreme prikazuje "
            "semanti\u010dku pretragu i TF-IDF. Dugme Pripremi BM25 dodaje tre\u0107i indeks bez "
            "u\u010ditavanja neuralnog modela i bez menjanja postoje\u0107ih izvora i vektora."
        )
        prepare_requested = st.button("Pripremi BM25", key="prepare_bm25")


def add_sources(sources: tuple[DocumentInput, ...]) -> None:
    try:
        with st.status("Provera nove grupe...", expanded=True) as status:
            ingest_batch(sources)
            st.write("Grupa je \u010ditljiva. U\u010ditavanje lokalnog modela i izgradnja sva tri indeksa...")
            result = store.add(sources, get_encoder(config))
            status.update(label="Obrada je zavr\u0161ena.", state="complete", expanded=False)
    except (ValueError, OSError, RuntimeError) as error:
        show_error(error, "Grupa nije dodata. Postoje\u0107a aktivna kolekcija je sa\u010duvana.")
        return
    clear_results()
    st.session_state["notice"] = (
        f"Dodato dokumenata: {len(result.added_ids)}."
        if result.changed else "Nema novih dokumenata; aktivni indeks nije menjan."
    )
    st.session_state["duplicates"] = result.duplicate_file_names
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
    st.rerun()


with st.sidebar:
    st.markdown("### Kolekcija")
    hint("PDF sa tekstom ili UTF-8 TXT. Dokumenti ostaju lokalno u projektnom skladi\u0161tu.")
    uploads = st.file_uploader(
        "Dodaj materijale", type=["pdf", "txt"], accept_multiple_files=True,
        key=f"uploads_{st.session_state.get('upload_version', 0)}",
    )
    if st.button("Dodaj i indeksiraj", disabled=not uploads, type="primary", width="stretch",
                 key="add_uploads"):
        add_sources(tuple(DocumentInput(upload.name, upload.getvalue()) for upload in uploads))
    hint("Nova grupa dopunjava kolekciju. Jedan neispravan fajl odbija celu grupu.")
    with st.expander("Priprema i odr\u017eavanje", key="maintenance"):
        prepared_manifest = default_manifest_path()
        hint(f"Pripremljeni ulaz: {prepared_manifest.name}")
        if st.button("Dodaj pripremljene materijale", width="stretch", key="add_prepared"):
            try:
                samples = load_manifest_inputs(prepared_manifest)
            except (ValueError, OSError) as error:
                show_error(error, "Pripremljeni manifest nije mogu\u0107e u\u010ditati.")
            else:
                add_sources(samples)
        if st.button("Obnovi indeks", disabled=not store.active.exists(),
                     width="stretch", key="rebuild_index"):
            try:
                with st.status("Obnova iz sa\u010duvanih izvora...", expanded=False) as status:
                    store.rebuild(get_encoder(config))
                    status.update(label="Sva tri indeksa su obnovljena.", state="complete")
            except (ValueError, OSError, RuntimeError) as error:
                show_error(error, "Obnova nije uspela. Prethodni paket nije prepisan.")
            else:
                clear_results()
                st.session_state["notice"] = "Sva tri indeksa su obnovljena."
                st.rerun()
        hint("Skenirani PDF bez tekstualnog sloja nije podr\u017ean.")
    with st.expander(f"Dokumenti i upozorenja ({document_count})", key="collection_details"):
        if index:
            for document in index.documents:
                st.text(document.title or document.file_name)
                hint(document.file_name)
                for warning in document.warnings:
                    st.text(warning)
        else:
            st.info("Kolekcija jo\u0161 nije indeksirana.")
    with st.expander("Model i verzija", key="model_details"):
        hint("Gotov model na CPU-u. Nema treniranja pri postavljanju pitanja.")
        st.code(config.model_name, language=None, wrap_lines=True)
        hint(f"Prozor {config.chunk_tokens} / preklapanje {config.overlap_tokens} tokena.")
        hint(
            f"BM25: k1={BM25_PARAMETERS['k1']}, b={BM25_PARAMETERS['b']}; pozitivan IDF, "
            "svaki termin upita doprinosi jednom. Parametri su fiksirani pre merenja."
        )
        if index:
            st.code(index.generation, language=None, wrap_lines=True)
    st.divider()
    details = st.toggle("Tehni\u010dki detalji rezultata", value=False, key="technical_details",
                        help="Prikazuje identifikatore i dodatne podatke; ne menja pretragu.")
    hint("Koristite samo materijale koje smete da obra\u0111ujete.")

with st.container(key="query_desk"):
    with st.form("search_form", border=False):
        query = st.text_input("Pitanje na engleskom", placeholder="What is regularization?",
                              key="query_input")
        method_label = st.radio(
            "Na\u010din pretrage", ["Pore\u0111enje", "Semanti\u010dka", "TF-IDF", "BM25"],
            horizontal=True, key="search_method",
            help="Pore\u0111enje koristi sve pripremljene metode. TF-IDF i BM25 ne u\u010ditavaju neuralni model.",
        )
        with st.container(key="search_actions"):
            actions = st.columns([2, 1], gap="small")
            with actions[0]:
                submitted = st.form_submit_button("Pretra\u017ei materijale", disabled=index is None,
                                                  type="primary", width="stretch", key="search_submit")
            with actions[1]:
                st.form_submit_button("O\u010disti upit", on_click=clear_search, width="stretch",
                                      key="clear_query", help="Ne bri\u0161e materijale ni indekse.")
    with st.container(horizontal=True, key="examples", gap="small"):
        for number, example in enumerate(EXAMPLES):
            st.button(example, type="tertiary", key=f"example_{number}",
                      on_click=use_example, args=(example,),
                      help="Popuni polje; pretragu pokre\u0107e dugme Pretra\u017ei materijale.")

if prepare_requested:
    # Render the search controls before rerunning so their widget state is retained.
    with app_notices:
        try:
            with st.status("Priprema BM25 iz sa\u010duvanih odlomaka...", expanded=False) as status:
                result = store.upgrade_bm25(config)
                status.update(label="BM25 je spreman.", state="complete")
        except (ValueError, OSError, RuntimeError) as error:
            show_error(error, "Priprema BM25 nije uspela. Prethodna kolekcija je sa\u010duvana.")
        else:
            clear_results()
            st.session_state["notice"] = (
                "BM25 je pripremljen. Izvori, odlomci i postoje\u0107i vektori nisu menjani."
                if result.changed else "BM25 je ve\u0107 pripremljen; indeks nije menjan."
            )
            st.rerun()

with st.container(key="search_feedback"):
    if submitted and index is not None:
        clear_results()
        methods = {
            "Pore\u0111enje": available_methods,
            "Semanti\u010dka": ("semantic",),
            "TF-IDF": ("tfidf",),
            "BM25": ("bm25",),
        }[method_label]
        try:
            if not query.strip():
                raise ValueError("Unesite neprazan upit na engleskom.")
            if "bm25" in methods and not bm25_available:
                raise ValueError("BM25 nije pripremljen. Koristite dugme Pripremi BM25 iznad pretrage.")
            with st.status("U\u010ditavanje i pretraga...", expanded=False) as status:
                encoder = get_encoder(config) if "semantic" in methods else None
                responses = []
                timings = {}
                for method in methods:
                    start = perf_counter_ns()
                    responses.append(search(index, query, method, encoder))
                    timings[method] = (perf_counter_ns() - start) / 1e6
                status.update(label="Rangiranje je zavr\u0161eno.", state="complete", expanded=False)
        except (ValueError, OSError, RuntimeError) as error:
            show_error(error, "Pretraga nije uspela.")
        else:
            st.session_state["responses"] = tuple(responses)
            st.session_state["search_timings"] = timings


def panel_header(method: str, response: SearchResponse | None) -> None:
    appearance = METHODS[method]
    status = "ZAVR\u0160ENO" if response else "\u010cEKA UPIT"
    state = "" if response else "waiting"
    st.html(
        f'<div class="method-header" style="--accent:{appearance["color"]}">'
        '<div class="method-name"><span class="method-dot"></span>'
        f'<h2>{appearance["name"]}</h2><span class="role-tag">{appearance["role"]}</span></div>'
        f'<span class="status-pill {state}">{status}</span></div>'
    )
    if response:
        elapsed = st.session_state.get("search_timings", {}).get(method)
        timing = (
            '<span title="Vreme ovog poziva; bez ucitavanja modela i prikaza. Nije benchmark.">'
            f'<strong>{elapsed:.1f} ms</strong> ovaj upit</span>'
            if elapsed is not None else "<span>vreme nije zabele\u017eeno</span>"
        )
        st.html(
            '<div class="method-stats">'
            f'<span><strong>{len(response.hits)}</strong> kandidata</span>'
            f'{timing}<span>TOP-{config.top_k}</span></div>'
        )
    else:
        st.html(
            '<div class="empty-panel"><strong>Spremno za pore\u0111enje.</strong>'
            f'{escape(appearance["description"])} Prvi rezultat bi\u0107e otvoren za \u010ditanje.</div>'
        )


def render_hit(hit: SearchHit, source: DocumentInfo, downloads: dict[str, bytes]) -> None:
    appearance = METHODS[hit.method]
    location = f"PDF \u00b7 strana {hit.page_start}" if hit.page_start is not None else "TXT \u00b7 bez numeracije strana"
    title = source.title or source.file_name.replace("_", " ")
    st.html(
        f'<div class="hit-meta" style="--accent:{appearance["color"]}">'
        f'<span class="rank-label">RANG {hit.rank:02d} / IZVORNI ODLOMAK</span>'
        f'<span class="score" title="Skor rangiranja unutar ove metode, ne verovatnoca tacnosti.">'
        f'SKOR {hit.score:.4f}</span></div>'
        f'<h3 class="hit-title">{escape(title)}</h3>'
        f'<div class="source-location">{escape(location)}</div>'
    )
    reader_tab, original_tab = st.tabs(["\u010citala\u010dki prikaz", "Original"],
                                       key=f"tabs_{hit.method}_{hit.chunk_id}")
    with reader_tab:
        view = reading_view(hit.text)
        if view.blocks:
            st.html(reading_html(view))
        else:
            st.info("Odlomak je prete\u017eno matemati\u010dki. Proverite izdvojene delove ili karticu Original.")
        if view.numeric_fragments or view.formula_fragments:
            label = "Formule i raspored" if view.formula_fragments else "Broj\u010dani delovi"
            with st.popover(f"{label} ({view.folded_lines} redova)", width="stretch",
                            key=f"numeric_{hit.method}_{hit.chunk_id}"):
                hint("Izdvojeni delovi izvornog rasporeda nisu obrisani. Formule nisu rekonstruisane; ta\u010dan oblik je u dokumentu.")
                for fragment in view.formula_fragments:
                    st.caption("Matemati\u010dki fragment \u00b7 sa izvornim prelomima")
                    st.code(fragment, language=None, wrap_lines=True, height=240)
                for fragment in view.numeric_fragments:
                    st.caption("Broj\u010dani fragment \u00b7 mogu\u0107a osa, tabela ili matrica")
                    st.code(fragment, language=None, wrap_lines=True, height=220)
        if view.replacement_characters:
            hint(f"Original sadr\u017ei {view.replacement_characters} neprepoznatih znakova. Za ta\u010dan tekst i formule proverite dokument.")
        hint("Prozni redovi su spojeni za \u010ditanje; formule se ne spajaju sa obja\u0161njenjem. Rang i skor ostaju izvorni.")
    with original_tab:
        st.code(hit.text, language=None, wrap_lines=True, height=300)
        st.text(source.file_name)
        if source.license_name:
            hint(f"Licenca izvora: {source.license_name}")
        if details:
            st.code(hit.chunk_id, language=None, wrap_lines=True)
            st.code(hit.document_id, language=None, wrap_lines=True)
    with st.container(horizontal=True, gap="small"):
        if source.source_url:
            try:
                parsed = urlsplit(source.source_url)
            except ValueError:
                parsed = None
            if parsed and parsed.scheme in {"http", "https"} and parsed.netloc:
                st.link_button("Otvori izvor", source.source_url)
            else:
                st.text("Link izvora nije validan HTTP/HTTPS URL.")
        content = downloads.get(source.sha256)
        if content is not None:
            st.download_button(
                "Preuzmi dokument", content, file_name=source.file_name,
                mime="application/pdf" if source.format == "pdf" else "text/plain; charset=utf-8",
                key=f"download_{hit.method}_{hit.chunk_id}", on_click="ignore",
            )
        else:
            hint("Originalna datoteka trenutno nije dostupna za preuzimanje.")


def show_response(response: SearchResponse, current: SearchIndex, downloads: dict[str, bytes]) -> None:
    panel_header(response.method, response)
    for warning in response.warnings:
        if warning.startswith("Nema poznatih termina"):
            st.warning(warning)
    documents = {document.document_id: document for document in current.documents}
    for hit in response.hits:
        source = documents[hit.document_id]
        if hit.rank == 1:
            with st.container(key=f"top_{hit.method}"):
                render_hit(hit, source, downloads)
        else:
            title = source.title or source.file_name.replace("_", " ")
            title = title if len(title) <= 65 else title[:62] + "\u2026"
            with st.expander(
                f"**{hit.rank:02d}** \u00b7 {markdown_label(title)}",
                key=f"result_{hit.method}_{hit.chunk_id}",
            ):
                render_hit(hit, source, downloads)


responses = st.session_state.get("responses", ())
with st.container(key="results_region"):
    if index is not None and responses:
        st.html(
            '<div class="results-heading"><div class="eyebrow">REZULTATI PRETRAGE</div>'
            f'<h2>Izvori za postavljeno pitanje</h2><div class="executed-query">{escape(responses[0].query)}</div>'
            '<div class="result-guidance">Ovo su prona\u0111eni odlomci, ne generisani odgovori. '
            'Skorovi razli\u010ditih metoda nisu na zajedni\u010dkoj skali i nisu procenat ta\u010dnosti.</div></div>'
        )
        try:
            downloads = get_downloads(index_directory, active_key)
        except (ValueError, OSError) as error:
            show_error(error, "Preuzimanje izvornih datoteka nije dostupno. Sa\u010duvani odlomci su prikazani ispod.")
            downloads = {}
        with st.container(key="results_grid"):
            columns = st.columns(len(responses), gap="medium")
            for column, response in zip(columns, responses, strict=True):
                with column, st.container(key=f"panel_{response.method}"):
                    show_response(response, index, downloads)
    elif index is not None:
        with st.container(key="results_grid"):
            columns = st.columns(len(available_methods), gap="medium")
            for column, method in zip(columns, available_methods, strict=True):
                with column, st.container(key=f"panel_{method}"):
                    panel_header(method, None)
    else:
        st.info("Prvo dodajte materijale kroz bo\u010dnu sekciju Kolekcija. Pretraga se uklju\u010duje posle indeksiranja.")

with st.expander("Sa\u010duvani eksperiment \u00b7 nije ocena ovog upita", key="saved_experiment"):
    if index is None:
        st.info("Za prikaz uporedivih rezultata prvo je potreban aktivni indeks.")
    elif not EXPERIMENT.exists():
        st.info("Nema sa\u010duvanih zavr\u0161nih rezultata eksperimenta za sve tri metode.")
    else:
        try:
            experiment = load_saved_experiment(EXPERIMENT, index)
        except (ValueError, OSError) as error:
            st.info(str(error))
        else:
            review_label = ("Ljudski pregled oznaka evidentiran."
                            if experiment.human_review_complete
                            else "AI-pregledane oznake; bez ljudske validacije.")
            hint(
                f"Zavr\u0161eno {experiment.measured_on} \u00b7 {experiment.query_count} test pitanja / "
                f"{experiment.intent_count} informacionih potreba. {review_label}"
            )
            metric_fields = [("hit_at_5", "Hit@5 \u00b7 relevantan odlomak u prvih pet"),
                             ("mrr_at_5", "MRR@5 \u00b7 rang prvog relevantnog odlomka")]
            if all(row.precision_at_5 is not None for row in experiment.rows):
                metric_fields.append(("precision_at_5", "Precision@5 \u00b7 udeo relevantnih me\u0111u pet rezultata"))
            else:
                hint("Precision@5 nije objavljen za ovaj skup: potreban je potpun pregled vra\u0107enih kandidata.")
            if experiment.evaluator_profile == "frozen_v1":
                hint("Istorijski eksperiment v1: proverena arhivska verzija evaluatora; nova evaluacija nije izvr\u0161ena.")
            for field, label in metric_fields:
                rows = []
                for row in experiment.rows:
                    appearance = METHODS[row.method]
                    value = getattr(row, field)
                    rows.append(
                        f'<div class="metric-row" style="--accent:{appearance["color"]}">'
                        f'<span>{"Semanti\u010dka" if row.method == "semantic" else appearance["name"]}</span>'
                        f'<div class="metric-track"><div class="metric-fill" style="width:{value * 100:.4f}%">'
                        f'</div></div><span>{value:.3f}</span></div>'
                    )
                st.html(f'<div class="metric-group"><div class="metric-title">{label}</div>{"".join(rows)}</div>')
            hint("Prikazane su samo sa\u010duvane metrike za ovu generaciju pretrage i proverenu verziju evaluatora. Nema \u017eivog progla\u0161avanja pobednika.")

st.html(
    '<footer class="desk-footer"><span>LOKALNO / CPU / PDF + TXT</span>'
    '<span>Izvor ostaje proverljiv. Model ne pi\u0161e novi odgovor.</span></footer>'
)
