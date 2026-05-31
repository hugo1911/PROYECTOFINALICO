
from reactpy import component, html, hooks
from reactpy.backend.starlette import configure
from starlette.applications import Starlette
from starlette.routing import Mount


@component
def StatusBadge():
    return html.span(
        {
            "style": {
                "display": "inline-block",
                "padding": "4px 12px",
                "borderRadius": "12px",
                "backgroundColor": "#22c55e",
                "color": "#fff",
                "fontSize": "0.8rem",
                "fontWeight": "600",
                "letterSpacing": "0.05em",
            }
        },
        "Backend: ONLINE",
    )


@component
def ChatPlaceholder():

    return html.div(
        {
            "style": {
                "border": "2px dashed #94a3b8",
                "borderRadius": "10px",
                "padding": "40px 20px",
                "textAlign": "center",
                "color": "#64748b",
                "marginTop": "24px",
                "backgroundColor": "#f8fafc",
            }
        },
        html.p({"style": {"margin": "0", "fontSize": "1rem"}},
               "Chat interface — not connected yet"),
        html.p({"style": {"margin": "8px 0 0", "fontSize": "0.85rem"}},
               "Will be wired to the RAG Assistant in the next step."),
    )


@component
def Layout():

    return html.div(
        {
            "style": {
                "fontFamily": "'Segoe UI', system-ui, sans-serif",
                "maxWidth": "860px",
                "margin": "0 auto",
                "padding": "32px 16px",
            }
        },
        html.header(
            {"style": {"borderBottom": "2px solid #e2e8f0", "paddingBottom": "16px", "marginBottom": "20px"}},
            html.h1(
                {"style": {"margin": "0 0 8px", "fontSize": "1.75rem", "color": "#1e293b"}},
                "Personal Digital Assistant",
            ),
            html.p(
                {"style": {"margin": "0 0 12px", "color": "#475569", "fontSize": "0.95rem"}},
                "RAG-powered chat — ask about emails, notes, SMS, and calendar.",
            ),
            StatusBadge(),
        ),
        html.main(
            {},
            ChatPlaceholder(),
        ),
        html.footer(
            {
                "style": {
                    "marginTop": "48px",
                    "paddingTop": "16px",
                    "borderTop": "1px solid #e2e8f0",
                    "color": "#94a3b8",
                    "fontSize": "0.8rem",
                    "textAlign": "center",
                }
            },
            "PROYECTOFINALICO — Inteligencia Computacional",
        ),
    )


app = Starlette()
configure(app, Layout)