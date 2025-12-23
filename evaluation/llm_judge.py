from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate


class LLMJudge:
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            Ты — система автоматической оценки качества ответов модели.

            Тебе даны:

            1. ВОПРОС пользователя
            2. ЭТАЛОННЫЙ ОТВЕТ (правильный)
            3. ОТВЕТ МОДЕЛИ

            Твоя задача — строго и объективно оценить ОТВЕТ МОДЕЛИ по шкале от 1 до 5.

            === ШКАЛА ОЦЕНКИ ===
            5 — Полностью корректный, точный, полный ответ, совпадает по смыслу с эталоном
            4 — В целом правильный, но есть небольшие неточности или упущены детали
            3 — Частично правильный, но есть серьёзные пропуски или формулировки расплывчаты
            2 — Слабое соответствие, затронута лишь малая часть правильного смысла
            1 — Ответ неверный, бессмысленный или не относится к вопросу

            === СТРОГИЕ ПРАВИЛА ===

            * Ты ОБЯЗАН различать качество: одинаковые ответы не должны ВСЕГДА получать один и тот же балл.
            * Если ответы похожи, но один чуть хуже — ты обязан понизить балл.
            * Если ответ формально похож, но теряет смысл — это 2 или 3, а НЕ 4–5.
            * Если информации недостаточно для уверенного ответа — не ставь 5.
            * Не будь избыточно доброжелательным.
            * Оценка должна быть обоснована.

            === ФОРМАТ ОТВЕТА (СТРОГО) ===
            Score: <число от 1 до 5>
            Reason: <краткое обоснование в 1–2 предложениях>

            === ДАННЫЕ ДЛЯ ОЦЕНКИ ===
            Вопрос:
            {question}

            Эталонный ответ:
            {reference_answer}

            Ответ модели:
            {model_answer}
            """
        )

    def judge_one(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
    ) -> Dict:
        final_prompt = self.prompt.format(
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
        )

        raw_response = self.llm.invoke(final_prompt).content.strip()

        try:
            score_line = [x for x in raw_response.split("\n") if "Score" in x][0]
            reason_line = [x for x in raw_response.split("\n") if "Reason" in x][0]

            score = float(score_line.split(":")[1].replace(",", ".").strip())
            reason = reason_line.split(":")[1].strip()

        except Exception:
            score = None
            reason = f"Parsing error: {raw_response}"

        return {
            "score": score,
            "reason": reason,
            "raw": raw_response,
        }

    def judge_batch(
        self,
        dataset: Dict[str, str],
        model_answers: List[str],
    ) -> Dict:
        results = []
        scores = []

        for i, question in enumerate(dataset.keys()):
            result = self.judge_one(
                question=question,
                reference_answer=dataset[question],
                model_answer=model_answers[i],
            )

            results.append({
                "question": question,
                "reference": dataset[question],
                "model_answer": model_answers[i],
                **result
            })

            if result["score"] is not None:
                scores.append(result["score"])

        return {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "scores": scores,
            "details": results,
        }
