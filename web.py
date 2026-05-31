import os
import glob as globmod
from reactpy import component, html, hooks
from reactpy.backend.starlette import configure
from starlette.applications import Starlette


DATA_DIR = "data/notes"


TYPE_RULES = [
    ("Codigo-Honor",        "Codigo de Honor"),
    ("Estacionamiento",     "Reglamento — Estacionamiento"),
    ("Practicas",           "Reglamento — Practicas Profesionales"),
    ("Servicio-Social",     "Reglamento — Servicio Social"),
    ("Titulacion",          "Reglamento — Titulacion"),
    ("Trimestral",          "Reglamento — Modalidad Trimestral"),
    ("Institucional",       "Reglamento — Institucional"),
    ("Estudiantes",         "Reglamento — Estudiantes Lic."),
]

ACCENT_MAP = str.maketrans(
    "aeiouAEIOUnN",
    "aeiouAEIOUnN",
)


def strip_accents(text: str) -> str:
    """Reemplaza vocales con acento por su version sin acento."""
    replacements = {
        "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
        "A": "A", "E": "E", "I": "I", "O": "O", "U": "U",
        "n": "n", "N": "N",
    }
    result = []
    for ch in text:
        result.append(replacements.get(ch, ch))
    return "".join(result)


def filename_to_label(filename: str) -> str:
    """Convierte CETYS_Reglamento-Foo-Bar-2024.txt en 'Reglamento Foo Bar 2024'."""
    name = filename.replace("CETYS_", "").replace(".txt", "")
    name = name.replace("-", " ")
    return strip_accents(name)


def detect_type(filename: str) -> str:
    for keyword, label in TYPE_RULES:
        if keyword in filename:
            return strip_accents(label)
    return "Documento"


def load_corpus_summary(data_dir: str = DATA_DIR) -> dict:
    """
    Escanea data_dir y devuelve:
      - total: numero de archivos .txt
      - types: conjunto de tipos detectados
      - docs: lista de dicts {name, type, size_kb}
    No instancia modelos ni llama al LLM.
    """
    pattern = os.path.join(data_dir, "*.txt")
    files = sorted(globmod.glob(pattern))

    docs = []
    types_seen = set()

    for path in files:
        fname = os.path.basename(path)
        size_kb = round(os.path.getsize(path) / 1024, 1)
        doc_type = detect_type(fname)
        label = filename_to_label(fname)
        types_seen.add(doc_type)
        docs.append({"name": label, "type": doc_type, "size_kb": size_kb, "path": path})

    return {
        "total": len(docs),
        "types": sorted(types_seen),
        "docs": docs,
    }


# CSS


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Georgia', 'Times New Roman', serif;
  background: #f4f1eb;
  color: #1a1a1a;
  min-height: 100vh;
}

.app-shell {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

/*  HEADER  */
.app-header {
  background: #1c2b3a;
  color: #e8e0d0;
  padding: 18px 32px;
  display: flex;
  align-items: baseline;
  gap: 20px;
  border-bottom: 3px solid #c8a96e;
}

.app-header h1 {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #e8e0d0;
}

.app-header .subtitle {
  font-size: 0.78rem;
  color: #8fa8b8;
  font-style: italic;
  letter-spacing: 0.04em;
}

.status-dot {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #7fc48a;
}

.status-dot::before {
  content: '';
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #7fc48a;
}

/*  BODY  */
.app-body {
  display: grid;
  grid-template-columns: 280px 1fr;
  overflow: hidden;
}

/*  SIDEBAR  */
.sidebar {
  background: #1e2d3d;
  color: #c8d8e4;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #2e4358;
  overflow-y: auto;
}

.sidebar-title {
  padding: 16px 20px 10px;
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #c8a96e;
  border-bottom: 1px solid #2e4358;
}

/*  Resumen de corpus  */
.corpus-summary {
  padding: 12px 20px;
  border-bottom: 1px solid #2e4358;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.summary-stat {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.72rem;
}

.summary-stat .label {
  color: #8fa8b8;
  letter-spacing: 0.04em;
}

.summary-stat .value {
  color: #dce8f0;
  font-weight: 700;
  font-size: 0.85rem;
}

.types-list {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.type-badge {
  font-size: 0.62rem;
  color: #7a9ab0;
  padding: 2px 0;
  letter-spacing: 0.03em;
  border-left: 2px solid #c8a96e;
  padding-left: 7px;
}

/*  Lista de documentos  */
.section-label {
  padding: 10px 20px 6px;
  font-size: 0.58rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #5a7a90;
  border-bottom: 1px solid #253548;
}

.doc-list {
  list-style: none;
  padding: 4px 0;
  flex: 1;
}

.doc-item {
  padding: 9px 20px;
  border-bottom: 1px solid #1a2838;
  cursor: default;
  transition: background 0.12s;
}

.doc-item:hover {
  background: #253d52;
}

.doc-name {
  font-size: 0.74rem;
  color: #dce8f0;
  line-height: 1.35;
}

.doc-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 3px;
}

.doc-type {
  font-size: 0.61rem;
  color: #7a9ab0;
  font-style: italic;
}

.doc-size {
  font-size: 0.61rem;
  color: #4a6070;
}

.corpus-footer {
  padding: 10px 20px;
  font-size: 0.62rem;
  color: #4a6070;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-top: 1px solid #2e4358;
}

/*  MAIN AREA  */
.main-area {
  background: #faf8f3;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  gap: 24px;
  overflow-y: auto;
}

.query-section-label {
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8a7a5e;
  margin-bottom: 6px;
}

.query-box {
  background: #fff;
  border: 1px solid #c8bfa8;
  border-radius: 2px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.query-box textarea {
  width: 100%;
  border: none;
  outline: none;
  font-family: 'Georgia', serif;
  font-size: 0.92rem;
  color: #1a1a1a;
  background: transparent;
  resize: none;
  line-height: 1.6;
  min-height: 80px;
}

.query-box textarea::placeholder {
  color: #b0a090;
  font-style: italic;
}

.query-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.query-hint {
  font-size: 0.7rem;
  color: #a09080;
  font-style: italic;
}

.btn-submit {
  background: #1c2b3a;
  color: #e8e0d0;
  border: none;
  padding: 9px 22px;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 1px;
  transition: background 0.2s;
  font-family: 'Georgia', serif;
}

.btn-submit:hover {
  background: #2e4358;
}

.btn-submit:disabled {
  background: #8a8170;
  cursor: not-allowed;
}

.btn-secondary {
  background: transparent;
  color: #6c5f49;
  border: 1px solid #c8bfa8;
  padding: 8px 14px;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 1px;
  transition: background 0.2s, color 0.2s;
  font-family: 'Georgia', serif;
}

.btn-secondary:hover {
  background: #f4f1eb;
  color: #1c2b3a;
}

.input-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.response-area {
  flex: 1;
  background: #fff;
  border: 1px solid #c8bfa8;
  border-radius: 2px;
  padding: 24px 28px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
}

.conversation-stream {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.message-row {
  display: flex;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant {
  justify-content: flex-start;
}

.chat-message {
  max-width: 72%;
  border: 1px solid #d8ccb3;
  padding: 12px 14px;
  line-height: 1.55;
  font-size: 0.88rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.message-row.user .chat-message {
  background: #1c2b3a;
  border-color: #1c2b3a;
  color: #e8e0d0;
}

.message-row.assistant .chat-message {
  background: #fbfaf7;
  color: #2d2922;
}

.message-label {
  display: block;
  margin-bottom: 5px;
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8a7a5e;
}

.message-row.user .message-label {
  color: #c8a96e;
}

.response-placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  text-align: center;
}

.response-placeholder .big-label {
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #c8bfa8;
}

.response-placeholder .desc {
  font-size: 0.82rem;
  font-style: italic;
  max-width: 320px;
  line-height: 1.6;
  color: #b0a090;
}

.divider-ornament {
  width: 40px;
  height: 1px;
  background: #c8bfa8;
  margin: 4px auto;
}

.sources-panel {
  margin-top: 18px;
  border-top: 1px solid #e0d6c4;
  padding-top: 14px;
}

.sources-title {
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8a7a5e;
  margin-bottom: 8px;
}

.sources-empty {
  font-size: 0.78rem;
  color: #b0a090;
  font-style: italic;
}

.source-list {
  list-style: none;
  display: grid;
  gap: 8px;
}

.source-item {
  border-left: 3px solid #c8a96e;
  background: #fbfaf7;
  padding: 8px 10px;
}

.source-name {
  display: block;
  font-size: 0.78rem;
  color: #1c2b3a;
  line-height: 1.35;
}

.source-score {
  display: block;
  margin-top: 2px;
  font-size: 0.62rem;
  color: #8a7a5e;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/*  FOOTER  */
.app-footer {
  background: #1c2b3a;
  color: #4a6070;
  font-size: 0.68rem;
  letter-spacing: 0.07em;
  text-align: center;
  padding: 10px 32px;
  border-top: 1px solid #2e4358;
  text-transform: uppercase;
}
"""

# COMPONENTS

@component
def StyleInjector():
    return html.style({}, CSS)


@component
def AppHeader():
    return html.header(
        {"className": "app-header"},
        html.h1({}, "Asistente RAG — CETYS"),
        html.span({"className": "subtitle"}, "Consulta de documentos institucionales"),
        html.span({"className": "status-dot"}, "Backend: online"),
    )


@component
def CorpusSummary(total, types):
    type_badges = [
        html.div({"className": "type-badge", "key": t}, t)
        for t in types
    ]
    return html.div(
        {"className": "corpus-summary"},
        html.div(
            {"className": "summary-stat"},
            html.span({"className": "label"}, "Total de documentos"),
            html.span({"className": "value"}, str(total)),
        ),
        html.div(
            {"className": "summary-stat"},
            html.span({"className": "label"}, "Tipos detectados"),
            html.span({"className": "value"}, str(len(types))),
        ),
        html.div({"className": "types-list"}, *type_badges),
    )


@component
def DocItem(name, doc_type, size_kb):
    return html.li(
        {"className": "doc-item"},
        html.div({"className": "doc-name"}, name),
        html.div(
            {"className": "doc-meta"},
            html.span({"className": "doc-type"}, doc_type),
            html.span({"className": "doc-size"}, f"{size_kb} KB"),
        ),
    )


@component
def Sidebar():
    summary, set_summary = hooks.use_state(None)

    @hooks.use_effect
    def fetch_corpus():
        data = load_corpus_summary()
        set_summary(data)

    if summary is None:
        return html.aside(
            {"className": "sidebar"},
            html.div({"className": "sidebar-title"}, "Corpus — Documentos"),
            html.div(
                {"style": {"padding": "20px", "fontSize": "0.75rem", "color": "#5a7a90"}},
                "Cargando corpus...",
            ),
        )

    doc_items = [
        DocItem(
            key=d["path"],
            name=d["name"],
            doc_type=d["type"],
            size_kb=d["size_kb"],
        )
        for d in summary["docs"]
    ]

    return html.aside(
        {"className": "sidebar"},
        html.div({"className": "sidebar-title"}, "Corpus — Documentos"),
        CorpusSummary(total=summary["total"], types=summary["types"]),
        html.div({"className": "section-label"}, "Reglamentos indexados"),
        html.ul({"className": "doc-list"}, *doc_items),
        html.div(
            {"className": "corpus-footer"},
            f"Directorio: {DATA_DIR}",
        ),
    )


def build_mock_answer(question: str) -> str:
    return (
        "Respuesta simulada: el backend RAG aun no esta conectado en esta vista. "
        "Cuando se integre, esta pregunta se enviara al recuperador y se respondera "
        f"con base en los documentos CETYS relacionados con: \"{question}\"."
    )


def build_mock_sources() -> list[dict[str, str]]:
    return [
        {
            "name": "Reglamento Institucional de Educacion Superior",
            "score": "simulado 0.91",
        },
        {
            "name": "Codigo de Honor Sistema CETYS",
            "score": "simulado 0.84",
        },
    ]


@component
def EmptyState():
    return html.div(
        {"className": "response-placeholder"},
        html.span({"className": "big-label"}, "Sin mensajes aun"),
        html.div({"className": "divider-ornament"}),
        html.p(
            {"className": "desc"},
            "Escribe una pregunta para ver aqui la conversacion simulada "
            "y las fuentes que recuperaria el asistente.",
        ),
    )


@component
def ChatMessage(role, content):
    label = "Usuario" if role == "user" else "Asistente"
    return html.div(
        {"className": f"message-row {role}"},
        html.div(
            {"className": "chat-message"},
            html.span({"className": "message-label"}, label),
            html.div({}, content),
        ),
    )


@component
def MessageList(messages):
    if not messages:
        return EmptyState()

    rendered_messages = [
        ChatMessage(
            key=f"{index}-{message['role']}",
            role=message["role"],
            content=message["content"],
        )
        for index, message in enumerate(messages)
    ]
    return html.div({"className": "conversation-stream"}, *rendered_messages)


@component
def SourcesPanel(sources):
    if not sources:
        return html.div(
            {"className": "sources-panel"},
            html.div({"className": "sources-title"}, "Fuentes recuperadas"),
            html.div(
                {"className": "sources-empty"},
                "Aun no hay fuentes para mostrar.",
            ),
        )

    source_items = [
        html.li(
            {"className": "source-item", "key": source["name"]},
            html.span({"className": "source-name"}, source["name"]),
            html.span({"className": "source-score"}, source["score"]),
        )
        for source in sources
    ]
    return html.div(
        {"className": "sources-panel"},
        html.div({"className": "sources-title"}, "Fuentes recuperadas"),
        html.ul({"className": "source-list"}, *source_items),
    )


@component
def InputControls(text, on_change, on_submit, on_clear, has_messages):
    return html.div(
        {"className": "query-box"},
        html.textarea(
            {
                "placeholder": "Escriba su pregunta sobre los reglamentos institucionales...",
                "value": text,
                "onChange": on_change,
                "rows": 4,
            }
        ),
        html.div(
            {"className": "query-footer"},
            html.span(
                {"className": "query-hint"},
                "Respuesta simulada hasta conectar el backend RAG",
            ),
            html.div(
                {"className": "input-actions"},
                html.button(
                    {
                        "className": "btn-secondary",
                        "onClick": on_clear,
                        "disabled": not has_messages and not text.strip(),
                    },
                    "Limpiar",
                ),
                html.button(
                    {
                        "className": "btn-submit",
                        "onClick": on_submit,
                        "disabled": not text.strip(),
                    },
                    "Enviar",
                ),
            ),
        ),
    )


@component
def QueryPanel():
    text, set_text = hooks.use_state("")
    messages, set_messages = hooks.use_state([])
    sources, set_sources = hooks.use_state([])

    def on_change(event):
        set_text(event["target"]["value"])

    def on_submit(event):
        question = text.strip()
        if not question:
            return

        answer = build_mock_answer(question)
        set_messages(
            messages
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        set_sources(build_mock_sources())
        set_text("")

    def on_clear(event):
        set_text("")
        set_messages([])
        set_sources([])

    return html.div(
        {"className": "main-area"},
        html.div(
            {},
            html.div({"className": "query-section-label"}, "Consulta"),
            InputControls(
                text=text,
                on_change=on_change,
                on_submit=on_submit,
                on_clear=on_clear,
                has_messages=bool(messages),
            ),
        ),
        html.div(
            {"style": {"flex": "1", "display": "flex", "flexDirection": "column"}},
            html.div({"className": "query-section-label"}, "Conversacion"),
            html.div(
                {"className": "response-area"},
                MessageList(messages=messages),
                SourcesPanel(sources=sources),
            ),
        ),
    )


@component
def AppFooter():
    return html.footer(
        {"className": "app-footer"},
        "PROYECTOFINALICO — Inteligencia Computacional — CETYS Universidad",
    )


@component
def Layout():
    return html.div(
        {"className": "app-shell"},
        StyleInjector(),
        AppHeader(),
        html.div(
            {"className": "app-body"},
            Sidebar(),
            QueryPanel(),
        ),
        AppFooter(),
    )


app = Starlette()
configure(app, Layout)
