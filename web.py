from reactpy import component, html, hooks
from reactpy.backend.starlette import configure
from starlette.applications import Starlette


CORPUS_DOCS = [
    ("Codigo de Honor", "Etica institucional"),
    ("Reglamento Estacionamiento 2021", "Accesos y regulacion"),
    ("Reglamento Lic. Escolarizada 2012-2013", "Normativa estudiantil"),
    ("Reglamento Lic. Escolarizada 2024-2025", "Normativa estudiantil"),
    ("Reglamento Lic. Escolarizada 2025-2026", "Normativa estudiantil"),
    ("Reglamento Lic. Trimestral 2025-2026", "Normativa estudiantil"),
    ("Reglamento Institucional Dic. 2024", "Educacion superior"),
    ("Reglamento Practicas Profesionales", "Vinculacion"),
    ("Reglamento Servicio Social Lic. 2024", "Servicio social"),
    ("Titulacion Automatica Lic. 2023", "Egreso"),
]

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
  grid-template-columns: 1fr;
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
  grid-template-columns: 260px 1fr;
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
  padding: 18px 20px 10px;
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #c8a96e;
  border-bottom: 1px solid #2e4358;
  font-family: 'Georgia', serif;
}

.doc-list {
  list-style: none;
  padding: 8px 0;
}

.doc-item {
  padding: 10px 20px;
  border-bottom: 1px solid #253548;
  cursor: default;
  transition: background 0.15s;
}

.doc-item:hover {
  background: #253d52;
}

.doc-name {
  font-size: 0.78rem;
  color: #dce8f0;
  line-height: 1.3;
}

.doc-category {
  font-size: 0.65rem;
  color: #8fa8b8;
  margin-top: 2px;
  font-style: italic;
}

.corpus-count {
  padding: 12px 20px;
  font-size: 0.65rem;
  color: #5a7a90;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: auto;
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
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8a7a5e;
  margin-bottom: 6px;
}

/*  INPUT BOX  */
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

/*  RESPONSE PLACEHOLDER  */
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

.response-placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #b0a090;
  text-align: center;
}

.response-placeholder .big-label {
  font-size: 0.7rem;
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
def SidebarDocItem(name, category):
    return html.li(
        {"className": "doc-item"},
        html.div({"className": "doc-name"}, name),
        html.div({"className": "doc-category"}, category),
    )


@component
def Sidebar():
    items = [SidebarDocItem(n, c) for n, c in CORPUS_DOCS]
    return html.aside(
        {"className": "sidebar"},
        html.div({"className": "sidebar-title"}, "Corpus — Documentos"),
        html.ul({"className": "doc-list"}, *items),
        html.div(
            {"className": "corpus-count"},
            f"{len(CORPUS_DOCS)} documentos indexados",
        ),
    )


@component
def QueryPanel():
    text, set_text = hooks.use_state("")

    def on_change(event):
        set_text(event["target"]["value"])

    def on_submit(event):
        pass  # backend connection — next step

    return html.div(
        {"className": "main-area"},
        html.div(
            {},
            html.div({"className": "query-section-label"}, "Consulta"),
            html.div(
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
                        "La consulta sera procesada por el motor RAG",
                    ),
                    html.button(
                        {"className": "btn-submit", "onClick": on_submit},
                        "Consultar",
                    ),
                ),
            ),
        ),
        html.div(
            {"style": {"flex": "1", "display": "flex", "flexDirection": "column"}},
            html.div({"className": "query-section-label"}, "Respuesta"),
            html.div(
                {"className": "response-area"},
                html.div(
                    {"className": "response-placeholder"},
                    html.span({"className": "big-label"}, "Sin respuesta aun"),
                    html.div({"className": "divider-ornament"}),
                    html.p(
                        {"className": "desc"},
                        "La respuesta del asistente aparecera aqui una vez "
                        "que se conecte el backend RAG en la siguiente etapa.",
                    ),
                ),
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
