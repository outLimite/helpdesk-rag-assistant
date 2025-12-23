from typing import List, Any, Optional
from rag.prompts import build_rag_prompt
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage, HumanMessage
import os

LLM_TYPE = os.environ.get("RAG_LLM", "llama-3.1-8b-instant").lower()


class Generator:
    def __init__(self, llm_name: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1028):
        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm_name is None:
            llm_name = LLM_TYPE


        self.llm = ChatGroq(model=llm_name, temperature=temperature, max_tokens=max_tokens, timeout=None, max_retries=2)

    def generate(self, question: str, retrieved_docs: List[Any]) -> str:
        prompt = build_rag_prompt(question, retrieved_docs)
        system = prompt["system"]
        user = prompt["user"]
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]

        response = self.llm.invoke(messages)
        return response.content
