import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.documents import Document
from rag import is_history_query, build_keyword_index, Assistant


DUMMY_CHUNK = Document(
    page_content="texto de prueba del reglamento",
    metadata={
        "path": "data/notes/CETYS_Reglamento-Dummy.txt",
        "document_type": "reglamento",
        "title": "Reglamento Dummy",
        "source_name": "CETYS_Reglamento-Dummy",
        "folder": "notes",
        "is_regulation": True,
    },
)


def make_assistant() -> Assistant:
    chunks = [DUMMY_CHUNK]
    index = build_keyword_index(chunks)
    return Assistant(
        index, None, chunks, None,
        {"embedding_model": "keyword", "llm_provider": "openai"},
    )


def run(label: str, got: bool, expected: bool, failures: list) -> None:
    ok = got == expected
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label!r} -> {got} (esperado {expected})")
    if not ok:
        failures.append(label)


def test_is_history_query_positivos(failures: list) -> None:
    casos = [
        "qué me acabas de preguntar",
        "que te acabo de preguntar",
        "qué me respondiste",
        "que me dijiste",
        "cuál fue tu respuesta",
        "cuál fue mi pregunta",
        "repite lo que dijiste",
        "repite lo que respondiste",
        "qué te pregunté",
        "lo que te pregunté antes",
        "de qué hemos hablado",
        "resume la conversacion",
        "mi última pregunta",
        "tu última respuesta",
        "lo que me respondiste",
        "lo que me dijiste",
    ]
    for q in casos:
        run(q, is_history_query(q), True, failures)


def test_is_history_query_negativos(failures: list) -> None:
    casos = [
        "cuántas faltas puedo tener",
        "qué dice el reglamento sobre titulación",
        "puedo reprobar materias",
        "cuál es el código de honor",
        "cuántos créditos necesito para titularme",
        "qué pasa si llego tarde",
        "cuáles son las sanciones disciplinarias",
        "en qué casos aplica el reglamento trimestral",
    ]
    for q in casos:
        run(q, is_history_query(q), False, failures)


def test_historial_vacio(failures: list) -> None:
    asistente = make_assistant()
    respuesta = asistente.ask("qué me respondiste")
    esperado = "No hay historial de conversacion todavia."
    ok = respuesta == esperado
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] historial vacio -> {respuesta!r}")
    if not ok:
        failures.append("historial_vacio")


def test_respuesta_desde_historial(failures: list) -> None:
    asistente = make_assistant()
    asistente.history = [
        {"role": "user", "content": "cuántas faltas puedo tener"},
        {"role": "assistant", "content": "Puedes tener máximo 3 faltas por materia."},
    ]

    mensajes_capturados: list = []

    def mock_llm(messages):
        mensajes_capturados.extend(messages)
        return "Tu pregunta anterior fue sobre el número de faltas."

    asistente._answer_from_llm = mock_llm
    respuesta = asistente.ask("qué me acabas de preguntar")

    historial_en_llm = any(
        m.get("content") == "cuántas faltas puedo tener" for m in mensajes_capturados
    )
    rag_en_llm = any("Contexto recuperado" in m.get("content", "") for m in mensajes_capturados)
    pregunta_guardada = asistente.history[-2]["content"] == "qué me acabas de preguntar"

    ok1, ok2, ok3 = historial_en_llm, not rag_en_llm, pregunta_guardada
    print(f"  [{'OK' if ok1 else 'FAIL'}] historial previo enviado al LLM")
    print(f"  [{'OK' if ok2 else 'FAIL'}] pregunta de historial NO paso por RAG")
    print(f"  [{'OK' if ok3 else 'FAIL'}] pregunta de historial guardada en historial")
    if not ok1:
        failures.append("historial_no_enviado_a_llm")
    if not ok2:
        failures.append("historia_paso_por_rag")
    if not ok3:
        failures.append("pregunta_historial_no_guardada")


def test_ask_with_sources_historial(failures: list) -> None:
    asistente = make_assistant()
    asistente.history = [
        {"role": "user", "content": "cuántos créditos necesito"},
        {"role": "assistant", "content": "Necesitas 240 créditos."},
    ]

    def mock_llm(messages):
        return "Tu pregunta fue sobre créditos."

    asistente._answer_from_llm = mock_llm
    _, fuentes = asistente.ask_with_sources("qué me respondiste")

    ok = fuentes == []
    print(f"  [{'OK' if ok else 'FAIL'}] ask_with_sources retorna fuentes vacías para pregunta de historial")
    if not ok:
        failures.append("ask_with_sources_fuentes_no_vacias")


def test_estrictez_reglamentos(failures: list) -> None:
    asistente = make_assistant()
    respuesta = asistente.ask("cuántas faltas puedo tener por semestre")
    ok = "No tengo suficiente informacion" in respuesta
    print(f"  [{'OK' if ok else 'FAIL'}] pregunta de reglamento sin contexto retorna mensaje sin info")
    if not ok:
        failures.append("reglamento_no_estricto")


def test_historial_se_acumula_correctamente(failures: list) -> None:
    asistente = make_assistant()

    def mock_llm(messages):
        return "respuesta_mock"

    asistente._answer_from_llm = mock_llm
    asistente.history = [
        {"role": "user", "content": "pregunta anterior"},
        {"role": "assistant", "content": "respuesta anterior"},
    ]

    asistente.ask("qué me respondiste")

    ok = len(asistente.history) == 4
    print(f"  [{'OK' if ok else 'FAIL'}] historial crece correctamente ({len(asistente.history)} entradas, esperado 4)")
    if not ok:
        failures.append("historial_no_crece")


if __name__ == "__main__":
    failures: list = []

    print("\n=== is_history_query: casos positivos ===")
    test_is_history_query_positivos(failures)

    print("\n=== is_history_query: casos negativos ===")
    test_is_history_query_negativos(failures)

    print("\n=== ask(): historial vacio ===")
    test_historial_vacio(failures)

    print("\n=== ask(): responde desde historial ===")
    test_respuesta_desde_historial(failures)

    print("\n=== ask_with_sources(): fuentes vacias en pregunta de historial ===")
    test_ask_with_sources_historial(failures)

    print("\n=== ask(): estrictez con reglamentos ===")
    test_estrictez_reglamentos(failures)

    print("\n=== ask(): historial se acumula correctamente ===")
    test_historial_se_acumula_correctamente(failures)

    print()
    if failures:
        print(f"FALLOS ({len(failures)}): {failures}")
        sys.exit(1)
    else:
        print("Todos los tests pasaron.")
