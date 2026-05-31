from __future__ import annotations

import os
import glob as globmod
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4

# Aqui conectamos cada carpeta con el tipo de documento que representa
DOCUMENT_FOLDERS = {
    "emails": "email",
    "notes": "note",
    "sms": "sms",
    "calendar": "calendar",
}

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"
ENV_TOP_K = "TOP_K"
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"


def load_config_from_env() -> dict[str, str | None]:
    """Aqui Cargamos la configuracion de las variables de entorno"""
    return {
        "api_key": os.getenv(ENV_OPENAI_API_KEY),
        "base_url": os.getenv(ENV_OPENAI_BASE_URL),
        "model": os.getenv(ENV_LLM_MODEL),
        "embedding_model": os.getenv(ENV_EMBEDDING_MODEL),
        "top_k": os.getenv(ENV_TOP_K),
        "chunk_size": os.getenv(ENV_CHUNK_SIZE),
        "chunk_overlap": os.getenv(ENV_CHUNK_OVERLAP),
    }


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed

#Si una variable no existe se asigna un valor por defecto
def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key") or None,
        "base_url": config.get("base_url") or None,
        "model": config.get("model") or DEFAULT_LLM_MODEL,
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


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    documents: list[Document] = []

    for folder_name, document_type in DOCUMENT_FOLDERS.items():
        # armamos la ruta para encontrar todos los .txt de esa carpeta
        pattern = os.path.join(data_dir, folder_name, "*.txt")

        for file_path in sorted(globmod.glob(pattern)):
            # leemos el archivo completo y lo guardamos como contenido del Document
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # guardamos tambien de donde salio y que tipo de documento es
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "path": file_path,
                        "document_type": document_type,
                    },
                )
            )

    return documents


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    # Aqui usamos los valores que llegaron de la configuracion, no numeros fijos.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # LangChain se encarga de partir el texto y mantener los metadatos en cada chunk.
    chunks = text_splitter.split_documents(docs)

    # Revisamos rapido que cada pedacito siga sabiendo de que archivo salio:p
    for chunk in chunks:
        if "path" not in chunk.metadata or "document_type" not in chunk.metadata:
            raise ValueError("Each chunk must preserve path and document_type metadata")

    return chunks


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    if not chunks:
        raise ValueError("Cannot build a FAISS index without chunks")

    texts = [chunk.page_content for chunk in chunks]

    # Agregamos el embedding model de la configutacion
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Como los embeddings ya vienen normalizados usamos producto interno
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    if not query.strip():
        return []

    if not chunks or index.ntotal == 0:
        return []

    search_k = min(k, len(chunks), index.ntotal)

    # Convertimos la pregunta en embedding para poder compararla contra FAISS
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

        # Regresamos lo q mas importa y es texto, score y metadatos del chunk encontrado
        results.append(
            {
                "text": chunk.page_content,
                "score": float(score),
                "metadata": chunk.metadata,
            }
        )

    return results


SYSTEM_PROMPT = (
    "Eres un asistente personal que responde usando solo el contexto recuperado "
    "de emails, notas, SMS y calendario. Si el contexto no alcanza para responder, "
    "debes decir que no tienes suficiente informacion."
)


def format_context(results: list[dict]) -> str:
    # Ponemos los chunks recuperados en un formato facil de meter al prompt.
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
    # Juntamos sistema, historial y pregunta actual con su contexto.
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


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        search_k = k or self.top_k

        # Antes de responder buscamos que pedazos de documentos tengan relacion con la pregunta
        relevant_chunks = retrieve(
            question,
            self.index,
            self.model,
            self.chunks,
            search_k,
        )

        if not relevant_chunks:
            return "No tengo suficiente información en los documentos para responder eso."

        context = format_context(relevant_chunks)
        messages = build_messages(question, context, self.history)

         # Aqui llamamos al LLM 
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )

        answer = completion.choices[0].message.content

        if not answer:
            answer = "No tengo suficiente información en los documentos para responder eso."



        # guardar historial conversacional
        # guardamos pregunta y respuesta para los siguientes turnos
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        return answer

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
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

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
