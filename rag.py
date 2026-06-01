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
import anthropic as anthropic_sdk

 DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4

 
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
    
        "llm_provider": (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower(),
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



NORMATIVE_KEYWORDS = [
    "Reglamento",
    "Codigo-Honor",
    "Codigo Honor",
    "Normativo",
    "Institucional",
]


def build_metadata(file_path: str, folder_name: str, document_type: str) -> dict:
    """Construye metadatos enriquecidos para un documento del corpus.

    Agrega title, source_name, folder, document_type e is_regulation
    a partir del nombre del archivo y la carpeta de origen.
    """
    filename = os.path.basename(file_path)
   
    title = filename.removesuffix(".txt").removeprefix("CETYS_").replace("-", " ").replace("_", " ")
  
    source_name = filename.removesuffix(".txt")
    
    is_regulation = any(kw in filename for kw in NORMATIVE_KEYWORDS)

    return {
        "path": file_path,
        "document_type": document_type,
        "title": title,
        "source_name": source_name,
        "folder": folder_name,
        "is_regulation": is_regulation,
    }


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Carga los .txt del corpus y les agrega metadatos basicos."""
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
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

     chunks = text_splitter.split_documents(docs)

     for chunk in chunks:
        if "path" not in chunk.metadata or "document_type" not in chunk.metadata:
            raise ValueError("Each chunk must preserve path and document_type metadata")

    return chunks


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Crea el indice FAISS con embeddings normalizados."""
    if not chunks:
        raise ValueError("Cannot build a FAISS index without chunks")

    texts = [chunk.page_content for chunk in chunks]

 
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
 
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
    """Busca los chunks mas cercanos a la pregunta."""
    if not query.strip():
        return []

    if not chunks or index.ntotal == 0:
        return []

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


def _call_anthropic(
    client: anthropic_sdk.Anthropic,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Llama al SDK oficial de Anthropic y regresa el texto de la respuesta.

    Extrae el system prompt del listado de mensajes para pasarlo como
    parametro separado, que es como lo espera la Messages API de Anthropic.
    Fallback: si la respuesta viene vacia regresa el mensaje de sin informacion.
    """
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
    if response.content and len(response.content) > 0:
        return response.content[0].text
    return "No tengo suficiente informacion en los reglamentos disponibles."


SYSTEM_PROMPT = (
    "Eres un asistente institucional de CETYS Universidad. Respondes usando solo "
    "el contexto recuperado de los reglamentos y documentos normativos de CETYS. "
    "Si el contexto no alcanza para responder, debes decir que no tienes "
    "suficiente informacion en los reglamentos disponibles."
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
            "title": title_from_path(path),
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
        self.llm_provider = self.config["llm_provider"]
        self.anthropic_client: anthropic_sdk.Anthropic | None = None
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        """Genera una respuesta usando contexto recuperado e historial."""
        search_k = k or self.top_k

     
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

    
        if self.llm_provider == "anthropic" and self.anthropic_client is not None:
            answer = _call_anthropic(self.anthropic_client, self.llm_model, messages)
        else:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
            )
            answer = completion.choices[0].message.content

        if not answer:
            answer = "No tengo suficiente información en los documentos para responder eso."



       
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        return answer

    def ask_with_sources(self, question: str, k: int | None = None) -> tuple[str, list[dict]]:
        """Responde y regresa las fuentes recuperadas para pintarlas en la UI."""
        search_k = k or self.top_k
        relevant_chunks = retrieve(question, self.index, self.model, self.chunks, search_k)

        if not relevant_chunks:
            return "No tengo suficiente información en los documentos para responder eso.", []

        context = format_context(relevant_chunks)
        msgs = build_messages(question, context, self.history)

 
        if self.llm_provider == "anthropic" and self.anthropic_client is not None:
            answer = _call_anthropic(self.anthropic_client, self.llm_model, msgs)
        else:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=msgs,
                temperature=0.2,
            )
            answer = completion.choices[0].message.content
        if not answer:
            answer = "No tengo suficiente información en los documentos para responder eso."

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer, unique_sources(relevant_chunks)

    def clear_history(self) -> None:
        """Limpia el historial de la conversacion."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Inicializa documentos, embeddings, FAISS y cliente LLM."""
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


        instance = cls(index, embedding_model, chunks, client, resolved_config)
        if resolved_config["llm_provider"] == "anthropic":
            anthropic_key = resolved_config.get("anthropic_api_key")
            if anthropic_key:
                instance.anthropic_client = anthropic_sdk.Anthropic(api_key=anthropic_key)
                instance.llm_model = resolved_config["model"] or DEFAULT_ANTHROPIC_MODEL
                print(f"  Proveedor: Anthropic Claude ({instance.llm_model})")
            else:
                print("  Advertencia: LLM_PROVIDER=anthropic pero ANTHROPIC_API_KEY no esta configurada.")
                print("  Fallback: OpenAI")
        else:
            print(f"  Proveedor: OpenAI ({instance.llm_model})")

        print("Ready!\n")
        return instance
