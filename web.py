from __future__ import annotations

import asyncio
import os
import glob as globmod
from reactpy import component, html, hooks
from reactpy.backend.starlette import configure
from starlette.applications import Starlette
from rag import Assistant, load_config_from_env


DATA_DIR = "data/notes"

try:
    _assistant: Assistant | None = Assistant.from_config(load_config_from_env())
    _init_error: str | None = None
except Exception as _e:
    _assistant = None
    _init_error = str(_e)


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


async def _run_query(
    question: str,
    msgs_snapshot: list,
    set_messages,
    set_sources,
    set_loading,
) -> None:
    """Manda la pregunta al RAG sin congelar la interfaz."""
    loop = asyncio.get_running_loop()
    try:
        if _assistant is None:
            raise RuntimeError(_init_error or "El asistente no pudo inicializarse.")
        answer, new_sources = await loop.run_in_executor(
            None, lambda: _assistant.ask_with_sources(question)
        )
        set_messages(msgs_snapshot + [{"role": "assistant", "content": answer}])
        set_sources(new_sources)
    except Exception as exc:
        set_messages(msgs_snapshot + [
            {"role": "error", "content": f"Error al consultar el asistente: {exc}"}
        ])
        set_sources([])
    finally:
        set_loading(False)


# Estilos de la interfaz.


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Georgia', 'Times New Roman', serif;
  background: #ffffff;
  color: #111111;
  min-height: 100vh;
}

.app-shell {
  display: grid;
  grid-template-rows: auto 1fr auto;
  height: 100vh;
  overflow: hidden;
}

/*  Encabezado  */
.app-header {
  background: #111111;
  color: #ffffff;
  padding: 14px 32px;
  display: flex;
  align-items: center;
  gap: 16px;
  border-bottom: 3px solid #FFD600;
  min-height: 0;
}

.app-header h1 {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #FFD600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.app-header .subtitle {
  font-size: 0.78rem;
  color: #cccccc;
  font-style: italic;
  letter-spacing: 0.04em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.status-dot {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #FFD600;
  white-space: nowrap;
  flex-shrink: 0;
}

.status-dot::before {
  content: '';
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #FFD600;
}

/*  Estructura principal  */
.app-body {
  display: grid;
  grid-template-columns: 260px 1fr;
  overflow: hidden;
  min-height: 0;
}

.app-body > * {
  min-width: 0;
}

/*  Barra lateral  */
.sidebar {
  background: #111111;
  color: #ffffff;
  display: flex;
  flex-direction: column;
  border-right: 3px solid #FFD600;
  overflow-y: auto;
  min-width: 0;
}

.sidebar-title {
  padding: 18px 20px 12px;
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #FFD600;
  border-bottom: 1px solid #333333;
  font-weight: 700;
}

/*  Resumen de corpus  */
.corpus-summary {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex: 1;
}

.summary-stat {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #1e1e1e;
  border-left: 3px solid #FFD600;
  padding: 10px 14px;
  border-radius: 2px;
}

.summary-stat .label {
  font-size: 0.70rem;
  color: #cccccc;
  letter-spacing: 0.04em;
}

.summary-stat .value {
  color: #FFD600;
  font-weight: 700;
  font-size: 1.1rem;
}

.types-list {
  margin-top: 4px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.types-list-label {
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #888888;
  margin-bottom: 4px;
}

.type-badge {
  font-size: 0.65rem;
  color: #eeeeee;
  padding: 5px 10px;
  letter-spacing: 0.03em;
  border-left: 2px solid #FFD600;
  background: #1a1a1a;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  border-radius: 1px;
}

/*  Area principal  */
.main-area {
  background: #f9f9f9;
  display: flex;
  flex-direction: column;
  padding: 28px 36px;
  gap: 20px;
  overflow-y: auto;
  min-width: 0;
}

.query-section-label {
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #555555;
  margin-bottom: 6px;
}

.query-box {
  background: #ffffff;
  border: 2px solid #FFD600;
  border-radius: 3px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: 0 2px 8px rgba(255,214,0,0.10);
}

.query-box textarea {
  width: 100%;
  border: none;
  outline: none;
  font-family: 'Georgia', serif;
  font-size: 0.92rem;
  color: #111111;
  background: transparent;
  resize: none;
  line-height: 1.6;
  min-height: 80px;
  overflow-wrap: break-word;
  word-break: break-word;
}

.query-box textarea::placeholder {
  color: #aaaaaa;
  font-style: italic;
}

.query-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}

.query-hint {
  font-size: 0.7rem;
  color: #888888;
  font-style: italic;
}

.btn-submit {
  background: #FFD600;
  color: #111111;
  border: none;
  padding: 9px 24px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.18s, transform 0.1s;
  font-family: 'Georgia', serif;
  white-space: nowrap;
}

.btn-submit:hover {
  background: #e6c200;
}

.btn-submit:active {
  transform: scale(0.98);
}

.btn-submit:disabled {
  background: #dddddd;
  color: #999999;
  cursor: not-allowed;
}

.btn-secondary {
  background: transparent;
  color: #555555;
  border: 1px solid #cccccc;
  padding: 8px 14px;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.18s, color 0.18s, border-color 0.18s;
  font-family: 'Georgia', serif;
  white-space: nowrap;
}

.btn-secondary:hover {
  background: #FFD600;
  color: #111111;
  border-color: #FFD600;
}

.btn-secondary:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.input-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.response-area {
  flex: 1;
  background: #ffffff;
  border: 1px solid #e8e8e8;
  border-top: 3px solid #FFD600;
  border-radius: 3px;
  padding: 24px 28px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow-y: auto;
}

.conversation-stream {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.message-row {
  display: flex;
  min-width: 0;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant {
  justify-content: flex-start;
}

.chat-message {
  max-width: 72%;
  min-width: 0;
  padding: 12px 16px;
  line-height: 1.55;
  font-size: 0.88rem;
  overflow-wrap: break-word;
  word-break: break-word;
  white-space: pre-wrap;
  border-radius: 3px;
}

.message-row.user .chat-message {
  background: #111111;
  color: #ffffff;
  border-left: 3px solid #FFD600;
}

.message-row.assistant .chat-message {
  background: #fffde6;
  color: #111111;
  border-left: 3px solid #FFD600;
  border: 1px solid #FFD600;
}

.message-label {
  display: block;
  margin-bottom: 5px;
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #888888;
  white-space: nowrap;
}

.message-row.user .message-label {
  color: #FFD600;
}

.message-row.assistant .message-label {
  color: #b89a00;
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
  color: #cccccc;
}

.response-placeholder .desc {
  font-size: 0.82rem;
  font-style: italic;
  max-width: 320px;
  line-height: 1.6;
  color: #aaaaaa;
}

.divider-ornament {
  width: 40px;
  height: 2px;
  background: #FFD600;
  margin: 4px auto;
  border-radius: 1px;
}

.sources-panel {
  margin-top: 18px;
  border-top: 1px solid #eeeeee;
  padding-top: 14px;
}

.sources-title {
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #555555;
  margin-bottom: 8px;
}

.sources-empty {
  font-size: 0.78rem;
  color: #aaaaaa;
  font-style: italic;
}

.source-list {
  list-style: none;
  display: grid;
  gap: 8px;
}

.source-item {
  border-left: 3px solid #FFD600;
  background: #fffde6;
  padding: 8px 12px;
  min-width: 0;
  border-radius: 0 2px 2px 0;
}

.source-title {
  display: block;
  font-size: 0.78rem;
  color: #111111;
  line-height: 1.35;
  font-weight: 700;
  overflow-wrap: break-word;
  word-break: break-word;
}

.source-meta {
  display: block;
  margin-top: 3px;
  font-size: 0.66rem;
  color: #555555;
  line-height: 1.35;
  overflow-wrap: break-word;
}

.source-file {
  display: block;
  margin-top: 2px;
  font-size: 0.62rem;
  color: #777777;
  word-break: break-all;
}

.source-score {
  display: block;
  margin-top: 2px;
  font-size: 0.62rem;
  color: #b89a00;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 700;
}

/*  Mensajes de error  */
.message-row.error .chat-message {
  background: #fff4f4;
  border-left: 3px solid #e05050;
  border: 1px solid #e8b0b0;
  color: #7a2020;
}

.message-row.error .message-label {
  color: #c05050;
}

/*  Indicador de busqueda  */
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.2; }
}

.searching-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #FFD600;
  margin-right: 8px;
  animation: blink 1.1s ease-in-out infinite;
}

/*  Pie de pagina  */
.app-footer {
  background: #111111;
  color: #666666;
  font-size: 0.68rem;
  letter-spacing: 0.07em;
  text-align: center;
  padding: 10px 32px;
  border-top: 2px solid #FFD600;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ===== RESPONSIVE ===== */

@media (max-width: 900px) {
  .app-body {
    grid-template-columns: 210px 1fr;
  }
  .main-area {
    padding: 20px 24px;
  }
  .response-area {
    padding: 18px 20px;
  }
}

@media (max-width: 640px) {
  .app-header {
    padding: 10px 16px;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 10px;
  }
  .app-header h1 {
    font-size: 0.95rem;
  }
  .app-header .subtitle {
    display: none;
  }
  .status-dot {
    margin-left: auto;
    font-size: 0.65rem;
  }
  .app-body {
    grid-template-columns: 1fr;
  }
  .sidebar {
    display: none;
  }
  .main-area {
    padding: 12px 14px;
    gap: 12px;
  }
  .response-area {
    padding: 12px 14px;
  }
  .chat-message {
    max-width: 92%;
    font-size: 0.85rem;
  }
  .query-footer {
    flex-direction: column;
    align-items: flex-start;
  }
  .input-actions {
    width: 100%;
    justify-content: flex-end;
  }
  .query-hint {
    font-size: 0.68rem;
  }
  .app-footer {
    padding: 8px 16px;
    font-size: 0.62rem;
  }
}
"""

# Componentes de ReactPy.

@component
def StyleInjector():
     return html._(
        html.meta({"name": "viewport", "content": "width=device-width, initial-scale=1.0"}),
        html.style({}, CSS),
    )


@component
def AppHeader():
    cetys_logo = html.svg(
        {
            "viewBox": "0 0 160 48",
            "width": "140",
            "height": "42",
            "xmlns": "http://www.w3.org/2000/svg",
            "style": {"marginLeft": "auto", "flexShrink": "0"},
        },
        html.rect({"x": "0", "y": "0", "width": "160", "height": "48", "rx": "3", "fill": "#FFD600"}),
        html.text(
            {"x": "12", "y": "30", "fontFamily": "Georgia, serif", "fontWeight": "900",
             "fontSize": "26", "fill": "#111111", "letterSpacing": "2"},
            "CETYS",
        ),
        html.text(
            {"x": "12", "y": "43", "fontFamily": "Georgia, serif", "fontWeight": "400",
             "fontSize": "10", "fill": "#333333", "letterSpacing": "3"},
            "UNIVERSIDAD",
        ),
        html.rect({"x": "108", "y": "8", "width": "3", "height": "32", "fill": "#111111"}),
        html.text(
            {"x": "116", "y": "24", "fontFamily": "Georgia, serif", "fontWeight": "700",
             "fontSize": "8", "fill": "#111111", "letterSpacing": "0.5"},
            "Asistente",
        ),
        html.text(
            {"x": "116", "y": "35", "fontFamily": "Georgia, serif", "fontWeight": "400",
             "fontSize": "8", "fill": "#333333", "letterSpacing": "0.5"},
            "RAG",
        ),
    )
    return html.header(
        {"className": "app-header"},
        html.h1({}, "Asistente RAG — CETYS"),
        cetys_logo,
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
        html.div(
            {"className": "types-list"},
            html.div({"className": "types-list-label"}, "Tipos en corpus"),
            *type_badges,
        ),
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
            html.div({"className": "sidebar-title"}, "Corpus"),
            html.div(
                {"style": {"padding": "20px", "fontSize": "0.75rem", "color": "#888888"}},
                "Cargando corpus...",
            ),
        )

    return html.aside(
        {"className": "sidebar"},
        html.div({"className": "sidebar-title"}, "Corpus"),
        CorpusSummary(total=summary["total"], types=summary["types"]),
    )



@component
def EmptyState():
    return html.div(
        {"className": "response-placeholder"},
        html.span({"className": "big-label"}, "Sin mensajes aun"),
        html.div({"className": "divider-ornament"}),
        html.p(
            {"className": "desc"},
            "Escribe una pregunta para consultar los reglamentos institucionales de CETYS.",
        ),
    )


@component
def ChatMessage(role, content):
    labels = {"user": "Usuario", "assistant": "Asistente", "error": "Error"}
    label = labels.get(role, role)
    return html.div(
        {"className": f"message-row {role}"},
        html.div(
            {"className": "chat-message"},
            html.span({"className": "message-label"}, label),
            html.div({}, content),
        ),
    )


@component
def SearchingMessage():
    return html.div(
        {"className": "message-row assistant"},
        html.div(
            {"className": "chat-message"},
            html.span({"className": "message-label"}, "Asistente"),
            html.div(
                {},
                html.span({"className": "searching-dot"}),
                "Buscando en documentos...",
            ),
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
            {"className": "source-item", "key": source["file"]},
            html.span({"className": "source-title"}, source["title"]),
            html.span(
                {"className": "source-meta"},
                f"Tipo de documento: {source['document_type']}",
            ),
            html.span({"className": "source-file"}, f"Archivo: {source['file']}"),
            html.span({"className": "source-score"}, f"Score: {source['score']}"),
        )
        for source in sources
    ]
    return html.div(
        {"className": "sources-panel"},
        html.div({"className": "sources-title"}, "Fuentes recuperadas"),
        html.ul({"className": "source-list"}, *source_items),
    )


@component
def InputControls(text, on_change, on_submit, on_clear, has_messages, loading):
    hint = "Buscando en documentos CETYS..." if loading else ""

    clear_disabled = loading or (not has_messages and not text.strip())

    submit_disabled = loading or not text.strip()

    return html.div(
        {"className": "query-box"},
        html.textarea(
            {
                "placeholder": "Escriba su pregunta sobre los reglamentos institucionales...",
                "value": text,
                "onChange": on_change,
                "rows": 4,
                "disabled": loading,
            }
        ),
        html.div(
            {"className": "query-footer"},
            html.span({"className": "query-hint"}, hint),
            html.div(
                {"className": "input-actions"},
                html.button(
                    {
                        "className": "btn-secondary",
                        "onClick": on_clear,
                        "disabled": clear_disabled,
                        "title": "Limpiar conversacion y reiniciar historial",
                    },
                    "Limpiar",
                ),
                html.button(
                    {
                        "className": "btn-submit",
                        "onClick": on_submit,
                        "disabled": submit_disabled,
                    },
                    "Buscando..." if loading else "Enviar",
                ),
            ),
        ),
    )


@component
def QueryPanel():
    text, set_text = hooks.use_state("")
    messages, set_messages = hooks.use_state([])
    sources, set_sources = hooks.use_state([])
    loading, set_loading = hooks.use_state(False)

    def on_change(event):
        set_text(event["target"]["value"])

    def on_submit(event):
        question = text.strip()
        if not question or loading:
            return

        user_msg = {"role": "user", "content": question}
        msgs_with_user = messages + [user_msg]
        set_text("")
        set_loading(True)
        set_sources([])
        set_messages(msgs_with_user)
        asyncio.get_event_loop().create_task(
            _run_query(question, msgs_with_user, set_messages, set_sources, set_loading)
        )

    def on_clear(event):
        if loading:
            return

        set_text("")
        set_messages([])
        set_sources([])
        
        if _assistant is not None:
            _assistant.clear_history()

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
                loading=loading,
            ),
        ),
        html.div(
            {"style": {"flex": "1", "display": "flex", "flexDirection": "column"}},
            html.div({"className": "query-section-label"}, "Conversacion"),
            html.div(
                {"className": "response-area"},
                MessageList(messages=messages),
                SearchingMessage() if loading else None,
                SourcesPanel(sources=sources),
            ),
        ),
    )


@component
def AppFooter():
    return html.footer({"className": "app-footer"})


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
