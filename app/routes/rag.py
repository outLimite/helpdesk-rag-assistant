from fastapi import APIRouter, HTTPException
from ..schemas import QuestionRequest, AnswerResponse, HealthResponse
from rag.pipeline import RAGPipeline

router = APIRouter()

rag_pipeline = RAGPipeline(db_path="data/db/faiss_index")


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        result = rag_pipeline.answer(
            question=request.question,
            k=request.k,
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )


@router.post("/ask_with_scores")
def ask_with_scores(request: QuestionRequest):
    try:
        result = rag_pipeline.answer_with_scores(
            question=request.question,
            k=request.k,
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}
