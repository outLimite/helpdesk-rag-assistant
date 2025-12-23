# 🧠 Helpdesk RAG Assistant

Retrieval-Augmented Generation (RAG) ассистент для helpdesk / knowledge-base сценариев.
Проект реализует **полный RAG-пайплайн**: от загрузки и очистки данных до семантического поиска, генерации ответов и оценки качества.

---

## ✨ Возможности

- 🔎 Семантический поиск по базе знаний (FAISS)
- 🧩 Модульный RAG-пайплайн (retrieval → prompt → generation)
- 🚀 FastAPI API с эндпоинтом `/ask`
- 📦 Чёткое разделение логики по слоям
- 📊 Метрики качества retrieval и generation
- 🐳 Поддержка Docker / Docker Compose

---

## 🗂 Структура проекта

```text
helpdesk-rag-assistant/
├── app/                     # API слой
│   ├── main.py              # FastAPI entrypoint
│   ├── routes/
│   │   └── rag.py           # /ask endpoint
│   └── schemas.py           # Pydantic модели
│
├── rag/                     # Вся логика RAG
│   ├── pipeline.py          # Полный RAG-пайплайн
│   ├── retriever.py         # FAISS поиск
│   ├── generator.py         # LLM генерация
│   └── prompts.py           # PromptTemplate
│
├── embeddings/
│   ├── model.py             # USER-bge-m3
│   └── embedder.py          # encode(), normalize()
│
├── vectorstore/
│   ├── faiss_store.py       # save/load/search
│   └── index_builder.py     # сборка индекса
│
├── ingestion/
│   ├── loader.py            # RecursiveUrlLoader
│   ├── cleaner.py           # regex очистка
│   └── splitter.py          # chunk_size, overlap
│
├── evaluation/
│   ├── uniformity.py        # метрика uniformity
│   └── llm_judge.py         # LLM-based оценка
│
├── data/
│   ├── raw/                 # сырые html / pkl
│   ├── processed/          # очищенные чанки
│   └── db/                  # FAISS index
│
├── scripts/
│   ├── load_data.py         # парсинг сайта
│   ├── build_index.py       # эмбеддинги + FAISS
│   └── evaluate.py          # метрики
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/outLimite/helpdesk-rag-assistant.git
cd helpdesk-rag-assistant
```

---

### 2. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Переменные окружения

Создай файл `.env`:

```env
LANGCHAIN_API_KEY=your_key
GROQ_API_KEY=your_key
```

---

### 4. Загрузка и подготовка данных

```bash
python scripts/load_data.py
python scripts/build_index.py
```

---

### 5. Запуск API

```bash
uvicorn app.main:app --reload
```

API будет доступен по адресу:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

---

## 📊 Оценка качества

- **Uniformity** — равномерность embedding-пространства
- **LLM Judge** — LLM-based оценка качества ответа

Запуск оценки:

```bash
python scripts/evaluate.py
```

---

## 🐳 Docker

Запуск через Docker Compose:

```bash
docker-compose up --build
```

---
