from fastapi import FastAPI

from .routes.rag import router as rag_router

app = FastAPI(
    title="Helpdesk RAG Assistant",
    description="AI-ассистент для поиска по базе знаний",
    version="1.0.0",
)

app.include_router(rag_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Helpdesk RAG Assistant API is running"}
