import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import cohere
import chromadb
from chromadb.config import Settings


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("Falta COHERE_API_KEY en el .env")

EMBEDDING_MODEL = "embed-multilingual-v2.0"
CHAT_MODEL = "command-r-plus-08-2024"


co = cohere.Client(api_key=COHERE_API_KEY)


chroma_client = chromadb.PersistentClient(
    path="./chroma_rag_api",
    settings=Settings(anonymized_telemetry=False)
)


collection = chroma_client.get_or_create_collection(
    name="api_rag_docs",
    metadata={"description": "Documentos del Challenge Semana 4"}
)


documents_db = {}

SIMILARITY_THRESHOLD = 0.3


BANNED_WORDS = ["insulto", "odio", "racista"] 


class UploadRequest(BaseModel):
    title: str
    content: str


class UploadResponse(BaseModel):
    message: str
    document_id: str


class GenerateEmbeddingsRequest(BaseModel):
    document_id: Optional[str] = None  # si es None, genero embeddings para todos


class MessageResponse(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    similarity_score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    context_used: Optional[str]
    similarity_score: Optional[float]
    grounded: bool



def simple_chunk(text: str, max_chars: int = 400) -> List[str]:
    """
    Chunking super simple: corto el texto en pedazos de max_chars.
    No es perfecto, pero para el challenge está bien.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def generate_embeddings_for_document(document_id: str):
    """
    Genera embeddings para un documento específico y los guarda en Chroma.
    Si el doc no existe, tiro error.
    """
    if document_id not in documents_db:
        raise ValueError(f"Documento {document_id} no existe")

    doc = documents_db[document_id]
    title = doc["title"]
    content = doc["content"]

    #Chunkear el contenido
    chunks = simple_chunk(content, max_chars=400)

    #Embeddings con Cohere (v2 -> input_type obligatorio)
    try:
        embed_response = co.embed(
            texts=chunks,
            model=EMBEDDING_MODEL,
            input_type="search_document"
        )
    except Exception as e:
        print("[ERROR] Falló co.embed en generate_embeddings_for_document:", e)
        raise RuntimeError("El servicio externo no pudo procesar la solicitud en este momento.")

    embeddings = embed_response.embeddings

    #Armo IDs para cada chunk
    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

    #Borro embeddings viejos de este documento (por si regenero)
    collection.delete(
        where={"document_id": document_id}
    )

    #Agrego embeddings nuevos a Chroma
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "title": title
            }
            for _ in chunks
        ]
    )

    print(f"[LOG] Generé {len(chunks)} chunks/embeddings para el documento {document_id}")


def search_similar_chunks(query: str, top_k: int = 3) -> List[SearchResult]:
    """
    Embebe la query, consulta Chroma y devuelve los mejores chunks.
    """
    
    try:
        embed_response = co.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="search_query"
        )
    except Exception as e:
        print("[ERROR] Falló co.embed en search_similar_chunks:", e)
        raise RuntimeError("El servicio externo no pudo procesar la solicitud en este momento.")

    query_embedding = embed_response.embeddings[0]

    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    
    distances = results["distances"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    search_results: List[SearchResult] = []
    for doc_text, meta, dist in zip(docs, metas, distances):
        similarity = 1 / (1 + dist)
        search_results.append(
            SearchResult(
                document_id=meta["document_id"],
                title=meta["title"],
                content_snippet=doc_text[:200] + ("..." if len(doc_text) > 200 else ""),
                similarity_score=similarity
            )
        )

    return search_results


def is_inappropriate(text: str) -> bool:
    """
    Filtro MUY básico para lenguaje inapropiado.
    """
    text_lower = text.lower()
    return any(bad_word in text_lower for bad_word in BANNED_WORDS)


#iniciofastapi

app = FastAPI(title="RAG API - Challenge Semana 4")


#ENDPOINTS

@app.post("/upload", response_model=UploadResponse)
def upload_document(req: UploadRequest):
    """
    Carga un nuevo documento en el sistema.
    Todavía no generamos embeddings, eso se hace en /generate-embeddings.
    """
    doc_id = str(uuid.uuid4())

    documents_db[doc_id] = {
        "title": req.title,
        "content": req.content
    }

    print(f"[LOG] Subí documento {doc_id} con título '{req.title}'")

    return UploadResponse(
        message="Document uploaded successfully",
        document_id=doc_id
    )


@app.post("/generate-embeddings", response_model=MessageResponse)
def generate_embeddings(req: GenerateEmbeddingsRequest):
    """
    Genera embeddings para un documento específico o para todos.
    """
    if not documents_db:
        raise HTTPException(status_code=400, detail="No hay documentos cargados.")

    try:
        if req.document_id:
            generate_embeddings_for_document(req.document_id)
            msg = f"Embeddings generated successfully for document {req.document_id}"
        else:
            # Genero embeddings para todos
            for doc_id in documents_db.keys():
                generate_embeddings_for_document(doc_id)
            msg = "Embeddings generated successfully for all documents"
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        
        return MessageResponse(message=str(e))

    return MessageResponse(message=msg)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Busca los documentos más relevantes para la consulta usando embeddings.
    """
    if not documents_db:
        raise HTTPException(status_code=400, detail="No hay documentos cargados.")
    try:
        results = search_similar_chunks(req.query)
    except RuntimeError as e:
       
        return SearchResponse(results=[])

    return SearchResponse(results=results)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Genera una respuesta usando los documentos relevantes.
    Acá aplico IA responsable: grounding, filtros y manejo de errores.
    """

    
    if is_inappropriate(req.question):
        return AskResponse(
            answer="No puedo responder a este tipo de consultas.",
            context_used=None,
            similarity_score=None,
            grounded=False
        )

    
    try:
        search_results = search_similar_chunks(req.question, top_k=1)
    except RuntimeError as e:
        return AskResponse(
            answer="El servicio externo no pudo procesar la solicitud en este momento.",
            context_used=None,
            similarity_score=None,
            grounded=False
        )

    if not search_results:
        return AskResponse(
            answer="No cuento con información suficiente para responder a esta consulta.",
            context_used=None,
            similarity_score=None,
            grounded=False
        )

    best = search_results[0]
    best_context = best.content_snippet
    best_score = best.similarity_score

    
    if best_score < SIMILARITY_THRESHOLD:
        return AskResponse(
            answer="No cuento con información suficiente para responder a esta consulta.",
            context_used=best_context,
            similarity_score=best_score,
            grounded=False
        )

  
    system_prompt = (
        "Sos un asistente que responde preguntas basándose EXCLUSIVAMENTE "
        "en el contexto que te paso. Si la respuesta no está en el contexto, "
        "decí: 'No cuento con información suficiente para responder a esta consulta.' "
        "Respondé en ESPAÑOL, de forma clara y breve."
    )

    user_prompt = f"Contexto:\n{best_context}\n\nPregunta del usuario:\n{req.question}"

    try:
        chat_resp = co.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer_text = chat_resp.message.content[0].text
    except Exception as e:
        print("[ERROR] Falló co.chat en /ask:", e)
        return AskResponse(
            answer="El servicio externo no pudo procesar la solicitud en este momento.",
            context_used=best_context,
            similarity_score=best_score,
            grounded=False
        )

    
    return AskResponse(
        answer=answer_text,
        context_used=best_context,
        similarity_score=best_score,
        grounded=True
    )





