# 🧠 RAG API – Challenge Semana 4
Este proyecto implementa una **API REST con FastAPI** que construye un sistema de  
**RAG (Retrieval Augmented Generation)** utilizando:

**Cohere** para embeddings y generación de respuestas
**ChromaDB** como vector store local
**FastAPI** para exponer los endpoints

El sistema permite cargar documentos de texto, generar embeddings, realizar búsquedas semánticas
y responder preguntas **únicamente usando el contexto recuperado** (grounded responses).

## 🚀 Tecnologías utilizadas
- Python 3.10+
- FastAPI
- Cohere API
- ChromaDB
- Pydantic
- python-dotenv
- Uvicorn

## 📦 Instalación y ejecución
### 1️⃣ Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_REPO>
