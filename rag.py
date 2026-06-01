from __future__ import annotations

import glob as globmod
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any

from langchain_core.documents import Document
from openai import OpenAI

DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4
KEYWORD_EMBEDDING_MODEL = "keyword"

DOCUMENT_FOLDERS = {
    "notes": "reglamento",
}

SOURCE_TYPE_RULES = [
    ("Codigo-Honor", "Codigo de Honor"),
    ("Estacionamiento", "Reglamento de Estacionamiento"),
    ("Practicas", "Reglamento de Practicas Profesionales"),
    ("Servicio-Social", "Reglamento de Servicio Social"),
    ("Titulacion", "Reglamento de Titulacion"),
    ("Trimestral", "Reglamento de Modalidad Trimestral"),
    ("Institucional", "Reglamento Institucional"),
    ("Estudiantes", "Reglamento de Estudiantes"),
]

NORMATIVE_KEYWORDS = [
    "Reglamento",
    "Codigo-Honor",
    "Codigo Honor",
    "Normativo",
    "Institucional",
]

HISTORY_QUERY_PATTERNS = [
    r"acab[ao]s?\s+de\s+preguntar",
    r"acab[oé]\s+de\s+preguntar",
    r"qu[eé]\s+(me\s+)?(respondiste|contestaste|dijiste)",
    r"cu[aá]l\s+fue\s+(tu\s+)?(respuesta|contestaci[oó]n)",
    r"cu[aá]l\s+fue\s+(mi\s+)?pregunta",
    r"(qu[eé]|lo\s+que)\s+(te\s+)?pregunt[eé]",
    r"qu[eé]\s+te\s+hab[ií]a\s+preguntado",
    r"repite\s+(lo\s+que\s+)?(dijiste|respondiste|contestaste)",
    r"de\s+qu[eé]\s+(hemos|estamos)\s+hablado",
    r"resum(e|ir?)\s+(la\s+)?(nuestra\s+)?conversaci[oó]n",
    r"qu[eé]\s+hemos\s+(estado\s+)?hablando",
    r"(mi|tu)\s+[uú]ltima\s+(pregunta|respuesta)",
    r"lo\s+que\s+(me\s+)?respondiste",
    r"lo\s+que\s+(me\s+)?dijiste",
]

STOPWORDS = {
    "a",
    "al",
    "ante",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "en",
    "entre",
    "es",
    "esa",
    "ese",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "no",
    "para",
    "por",
    "que",
    "se",
    "si",
    "sobre",
    "su",
    "sus",
    "un",
    "una",
    "y",
}

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"
ENV_TOP_K = "TOP_K"
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class KeywordIndex:
    """Indice lexical chiquito para arrancar rapido sin embeddings locales."""

    def __init__(
        self,
        inverted_index: dict[str, dict[int, int]],
        doc_lengths: list[int],
    ) -> None:
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.ntotal = len(doc_lengths)
        self.d = 0


def load_config_from_env() -> dict[str, str | None]:
    """Carga la configuracion desde variables de entorno."""
    return {
        "api_key": os.getenv(ENV_OPENAI_API_KEY),
        "base_url": os.getenv(ENV_OPENAI_BASE_URL),
        "model": os.getenv(ENV_LLM_MODEL),
        "embedding_model": os.getenv(ENV_EMBEDDING_MODEL),
        "top_k": os.getenv(ENV_TOP_K),
        "chunk_size": os.getenv(ENV_CHUNK_SIZE),
        "chunk_overlap": os.getenv(ENV_CHUNK_OVERLAP),
        "llm_provider": os.getenv(ENV_LLM_PROVIDER),
        "anthropic_api_key": os.getenv(ENV_ANTHROPIC_API_KEY),
    }


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resuelve la configuracion final con tipos y valores por defecto."""
    config = config or {}
    llm_provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower()
    default_model = (
        DEFAULT_ANTHROPIC_MODEL if llm_provider == "anthropic" else DEFAULT_LLM_MODEL
    )

    resolved = {
        "api_key": config.get("api_key") or None,
        "base_url": config.get("base_url") or None,
        "model": config.get("model") or default_model,
        "embedding_model": config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
        "top_k": _parse_int_setting(
            ENV_TOP_K,
            config.get("top_k") or DEFAULT_TOP_K,
        ),
        "chunk_size": _parse_int_setting(
            ENV_CHUNK_SIZE,
            config.get("chunk_size") or DEFAULT_CHUNK_SIZE,
        ),
        "chunk_overlap": _parse_int_setting(
            ENV_CHUNK_OVERLAP,
            config.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP,
        ),
        "llm_provider": llm_provider,
        "anthropic_api_key": config.get("anthropic_api_key") or None,
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved


def build_metadata(file_path: str, folder_name: str, document_type: str) -> dict:
    """Construye metadatos enriquecidos para un documento del corpus."""
    filename = os.path.basename(file_path)
    title = filename.removesuffix(".txt").removeprefix("CETYS_")
    title = title.replace("-", " ").replace("_", " ")
    source_name = filename.removesuffix(".txt")
    is_regulation = any(keyword in filename for keyword in NORMATIVE_KEYWORDS)

    return {
        "path": file_path,
        "document_type": document_type,
        "title": title,
        "source_name": source_name,
        "folder": folder_name,
        "is_regulation": is_regulation,
    }


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Carga los .txt de reglamentos y les agrega metadatos basicos."""
    documents: list[Document] = []

    for folder_name, document_type in DOCUMENT_FOLDERS.items():
        pattern = os.path.join(data_dir, folder_name, "*.txt")

        for file_path in sorted(globmod.glob(pattern)):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append(
                Document(
                    page_content=text,
                    metadata=build_metadata(file_path, folder_name, document_type),
                )
            )

    return documents


def split_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Parte documentos en chunks con traslape conservando metadatos."""
    chunks: list[Document] = []

    for doc in docs:
        text = doc.page_content
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(doc.metadata),
                    )
                )

            if end == len(text):
                break
            start = max(end - chunk_overlap, start + 1)

    for chunk in chunks:
        if "path" not in chunk.metadata or "document_type" not in chunk.metadata:
            raise ValueError("Each chunk must preserve path and document_type metadata")

    return chunks


def tokenize(text: str) -> list[str]:
    """Tokeniza texto en palabras utiles para busqueda lexical."""
    tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def is_history_query(question: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in HISTORY_QUERY_PATTERNS)


def build_keyword_index(chunks: list[Document]) -> KeywordIndex:
    """Arma un indice invertido simple para consultas por palabras clave."""
    if not chunks:
        raise ValueError("Cannot build a keyword index without chunks")

    inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
    doc_lengths: list[int] = []

    for chunk_index, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk.page_content))
        doc_lengths.append(sum(counts.values()))

        for token, count in counts.items():
            inverted_index[token][chunk_index] = count

    return KeywordIndex(dict(inverted_index), doc_lengths)


def build_vector_index(chunks: list[Document], embedding_model: Any) -> Any:
    """Crea el indice FAISS cuando si usamos embeddings locales."""
    if not chunks:
        raise ValueError("Cannot build a FAISS index without chunks")

    import faiss
    import numpy as np

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def build_index(chunks: list[Document], embedding_model: Any) -> Any:
    """Compatibilidad: arma el indice vectorial de siempre."""
    return build_vector_index(chunks, embedding_model)


def keyword_retrieve(
    query: str,
    index: KeywordIndex,
    chunks: list[Document],
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Recupera chunks por BM25 ligero sin cargar modelos de embeddings."""
    query_terms = tokenize(query)
    if not query_terms or not chunks or index.ntotal == 0:
        return []

    avg_doc_length = sum(index.doc_lengths) / max(index.ntotal, 1)
    query_counts = Counter(query_terms)
    scores: dict[int, float] = defaultdict(float)
    k1 = 1.2
    b = 0.75

    for term, query_weight in query_counts.items():
        postings = index.inverted_index.get(term)
        if not postings:
            continue

        doc_frequency = len(postings)
        idf = math.log(1 + (index.ntotal - doc_frequency + 0.5) / (doc_frequency + 0.5))

        for chunk_index, term_frequency in postings.items():
            doc_length = index.doc_lengths[chunk_index] or 1
            denominator = term_frequency + k1 * (
                1 - b + b * doc_length / max(avg_doc_length, 1)
            )
            bm25 = idf * (term_frequency * (k1 + 1)) / denominator
            scores[chunk_index] += bm25 * query_weight

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results: list[dict] = []

    for chunk_index, score in ranked[: min(k, len(ranked))]:
        chunk = chunks[chunk_index]
        results.append(
            {
                "text": chunk.page_content,
                "score": float(score),
                "metadata": chunk.metadata,
            }
        )

    return results


def vector_retrieve(
    query: str,
    index: Any,
    model: Any,
    chunks: list[Document],
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Busca los chunks mas cercanos con embeddings."""
    if not query.strip():
        return []
    if not chunks or index.ntotal == 0:
        return []

    import numpy as np

    search_k = min(k, len(chunks), index.ntotal)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, search_k)
    results: list[dict] = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "text": chunk.page_content,
                "score": float(score),
                "metadata": chunk.metadata,
            }
        )

    return results


def retrieve(
    query: str,
    index: Any,
    model: Any,
    chunks: list[Document],
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Despacha al retrieval lexical o vectorial segun el indice."""
    if isinstance(index, KeywordIndex):
        return keyword_retrieve(query, index, chunks, k)
    return vector_retrieve(query, index, model, chunks, k)


def _call_anthropic(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Llama a Anthropic separando el system prompt como pide su API."""
    system_content = ""
    user_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_content,
        messages=user_messages,
    )
    if response.content:
        return response.content[0].text
    return "No tengo suficiente informacion en los reglamentos disponibles."


SYSTEM_PROMPT = (
    "Eres un asistente institucional de CETYS Universidad. Respondes usando solo "
    "el contexto recuperado de los reglamentos y documentos normativos de CETYS. "
    "Si el contexto no alcanza para responder, debes decir que no tienes "
    "suficiente informacion en los reglamentos disponibles."
)

HISTORY_SYSTEM_PROMPT = (
    "Eres un asistente institucional de CETYS Universidad. "
    "El usuario te esta preguntando sobre el historial de esta conversacion. "
    "Responde basandote unicamente en lo que ya se ha dicho en la conversacion actual."
)


def format_context(results: list[dict]) -> str:
    context_parts = []

    for result in results:
        metadata = result["metadata"]
        context_parts.append(
            f"Fuente: {metadata['document_type']} - {metadata['path']}\n"
            f"Score: {result['score']:.3f}\n"
            f"Texto:\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_messages(
    question: str,
    context: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Contexto recuperado:\n{context}\n\n"
                f"Pregunta actual:\n{question}"
            ),
        }
    )

    return messages


def build_history_messages(
    question: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": HISTORY_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def title_from_path(path: str) -> str:
    """Convierte el nombre del archivo en un titulo mas amable."""
    filename = os.path.basename(path)
    title = filename.removesuffix(".txt").removeprefix("CETYS_")
    return title.replace("-", " ").replace("_", " ")


def source_type_from_path(path: str, fallback: str) -> str:
    """Saca un tipo legible desde el nombre del archivo."""
    filename = os.path.basename(path)
    for keyword, label in SOURCE_TYPE_RULES:
        if keyword in filename:
            return label
    return fallback


def unique_sources(results: list[dict]) -> list[dict[str, str]]:
    """Agrupa chunks repetidos para mostrar cada archivo una sola vez."""
    sources_by_path: dict[str, dict[str, str | float]] = {}

    for result in results:
        metadata = result["metadata"]
        path = metadata["path"]
        score = float(result["score"])
        current = sources_by_path.get(path)

        if current is not None and score <= float(current["raw_score"]):
            continue

        sources_by_path[path] = {
            "title": metadata.get("title") or title_from_path(path),
            "document_type": source_type_from_path(path, metadata["document_type"]),
            "file": os.path.basename(path),
            "score": f"{score:.3f}",
            "raw_score": score,
        }

    ordered_sources = sorted(
        sources_by_path.values(),
        key=lambda source: float(source["raw_score"]),
        reverse=True,
    )

    return [
        {
            "title": str(source["title"]),
            "document_type": str(source["document_type"]),
            "file": str(source["file"]),
            "score": str(source["score"]),
        }
        for source in ordered_sources
    ]


class Assistant:
    """Asistente RAG con indice, modelo, cliente LLM e historial."""

    def __init__(
        self,
        index: Any,
        model: Any,
        chunks: list[Document],
        client: OpenAI | None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.llm_provider = self.config["llm_provider"]
        self.anthropic_client: Any | None = None
        self.history: list[dict[str, str]] = []

    def _answer_from_llm(self, messages: list[dict[str, str]]) -> str:
        if self.llm_provider == "anthropic" and self.anthropic_client is not None:
            return _call_anthropic(self.anthropic_client, self.llm_model, messages)

        if self.client is None:
            raise RuntimeError("No hay cliente LLM configurado.")

        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""

    def _retrieve_context(self, question: str, k: int | None = None) -> list[dict]:
        search_k = k or self.top_k
        return retrieve(question, self.index, self.model, self.chunks, search_k)

    def ask(self, question: str, k: int | None = None) -> str:
        """Genera una respuesta usando contexto recuperado e historial."""
        if is_history_query(question):
            if not self.history:
                answer = "No hay historial de conversacion todavia."
            else:
                messages = build_history_messages(question, self.history)
                answer = self._answer_from_llm(messages)
                if not answer:
                    answer = "No pude recuperar el historial de la conversacion."
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            return answer

        relevant_chunks = self._retrieve_context(question, k)

        if not relevant_chunks:
            return "No tengo suficiente informacion en los reglamentos disponibles."

        context = format_context(relevant_chunks)
        messages = build_messages(question, context, self.history)
        answer = self._answer_from_llm(messages)

        if not answer:
            answer = "No tengo suficiente informacion en los reglamentos disponibles."

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def ask_with_sources(self, question: str, k: int | None = None) -> tuple[str, list[dict]]:
        """Responde y regresa las fuentes recuperadas para pintarlas en la UI."""
        if is_history_query(question):
            if not self.history:
                answer = "No hay historial de conversacion todavia."
            else:
                messages = build_history_messages(question, self.history)
                answer = self._answer_from_llm(messages)
                if not answer:
                    answer = "No pude recuperar el historial de la conversacion."
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            return answer, []

        relevant_chunks = self._retrieve_context(question, k)

        if not relevant_chunks:
            return "No tengo suficiente informacion en los reglamentos disponibles.", []

        context = format_context(relevant_chunks)
        messages = build_messages(question, context, self.history)
        answer = self._answer_from_llm(messages)

        if not answer:
            answer = "No tengo suficiente informacion en los reglamentos disponibles."

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer, unique_sources(relevant_chunks)

    def clear_history(self) -> None:
        """Limpia el historial de la conversacion."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Inicializa documentos, indice de retrieval y cliente LLM."""
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_name = str(resolved_config["embedding_model"]).lower()
        if embedding_name == KEYWORD_EMBEDDING_MODEL:
            print("Building keyword index...")
            embedding_model = None
            index = build_keyword_index(chunks)
            print(f"  Indexed {index.ntotal} chunks with lexical retrieval")
        else:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {resolved_config['embedding_model']}")
            embedding_model = SentenceTransformer(resolved_config["embedding_model"])

            print("Building FAISS index...")
            index = build_vector_index(chunks, embedding_model)
            print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client: OpenAI | None = None
        anthropic_client: Any | None = None

        if resolved_config["llm_provider"] == "anthropic":
            anthropic_key = resolved_config.get("anthropic_api_key")
            if not anthropic_key:
                raise RuntimeError(
                    "LLM_PROVIDER=anthropic pero ANTHROPIC_API_KEY no esta configurada."
                )
            import anthropic as anthropic_sdk

            anthropic_client = anthropic_sdk.Anthropic(api_key=anthropic_key)
            print(f"  Proveedor: Anthropic Claude ({resolved_config['model']})")
        else:
            client_kwargs = {}
            if resolved_config["api_key"]:
                client_kwargs["api_key"] = resolved_config["api_key"]
            if resolved_config["base_url"]:
                client_kwargs["base_url"] = resolved_config["base_url"]
            client = OpenAI(**client_kwargs)
            print(f"  Proveedor: OpenAI ({resolved_config['model']})")

        instance = cls(index, embedding_model, chunks, client, resolved_config)
        instance.anthropic_client = anthropic_client

        print("Ready!\n")
        return instance
