
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PDFPlumberLoader
import requests
from config import JSON_PATH, INDEX_PATH, AUTH_KEY, DOCS_PATH
import re
import pickle
import time
import json


### RAG система

llm = GigaChat(
    credentials=AUTH_KEY,
    model="GigaChat",
    temperature=0.3,
    verify_ssl_certs=False
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# PDF данные

loader = PyPDFDirectoryLoader(JSON_PATH)
print("Начинаем загрузку PDF...")
t0 = time.time()
docs = []
if os.path.exists(DOCS_PATH):
    print("Загружаем docs из pickle")
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
else:
    for file in os.listdir(JSON_PATH):
        if file.endswith(".pdf"):
            loader = PDFPlumberLoader(os.path.join(JSON_PATH, file))
            docs.extend(loader.load())
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

data_time = time.time() - t0
print("Загрузка завершена, страниц:", len(docs))
print("Время обработки:", data_time, "сек")



# Очистка
print("Очистка")
for doc in docs:
    text = doc.page_content
    text = re.sub(r"\s+", " ", text)
    doc.page_content = text.strip()


def is_useful_page(text: str) -> bool:
    bad_keywords = [
        "table of contents",
        "report of independent",
        "consolidated statements",
        "balance sheets"
    ]
    t = text.lower()
    return not any(k in t for k in bad_keywords)


docs = [d for d in docs if is_useful_page(d.page_content)]

# Разбиение
print("Разбиение")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

splits = text_splitter.split_documents(docs)

# Векторное хранилище

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if os.path.exists(INDEX_PATH):
    # Загружаем существующий индекс
    print("Загрузка из файла")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    # Создаём новый индекс из документов
    print("Создание нового индекса")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_PATH)
print("Ретриевер")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 50,
        "lambda_mult": 0.5
    }
)

def normalize_question(q):
    q = q.lower()
    q = q.replace("did ", "")
    q = q.replace(" mention ", " ")
    q = q.replace("?", "")
    return q

def retrieve_with_sources(question):
    docs = retriever.invoke(normalize_question(question))

    print("\n--- RETRIEVED CHUNKS ---")

    numbered_chunks = []
    for i, d in enumerate(docs):
        print(f"[{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}")
        print(d.page_content[:300])
        print("----")

        numbered_chunks.append(
            f"[{i}]\n{d.page_content}"
        )

    context = "\n\n".join(numbered_chunks)

    return context, docs


def answer_question(question_text, kind):
    context, docs = retrieve_with_sources(question_text)

    response = llm.invoke(
        prompt.format(
            context=context,
            question=question_text,
            kind=kind
        )
    )

    raw_output = response.strip()

    print("\nRAW MODEL OUTPUT:")
    print(raw_output)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Ошибка парсинга JSON")
        parsed = {"value": "N/A", "chunk_id": None}

    value = parsed.get("value", "N/A")
    chunk_id = parsed.get("chunk_id")

    # Валидация chunk_id
    if value == "N/A":
        chunk_id = None
    else:
        if chunk_id is None or not isinstance(chunk_id, int) or chunk_id >= len(docs):
            print("Некорректный chunk_id → принудительно N/A")
            value = "N/A"
            chunk_id = None

    # Формируем references только из выбранного чанка
    references = []

    if chunk_id is not None:
        chosen_doc = docs[chunk_id]
        references.append({
            "pdf_sha1": os.path.basename(chosen_doc.metadata.get("source", "unknown")),
            "page_index": chosen_doc.metadata.get("page", -1)
        })

    return {
        "question": question_text,
        "value": value,
        "references": references
    }
# Промпт

prompt = ChatPromptTemplate.from_template("""
Ты — ассистент по извлечению данных из документов.

Ответь на вопрос, используя ТОЛЬКО один фрагмент контекста.

Верни ответ СТРОГО в JSON формате:
{{ "value": ответ, "chunk_id": номер_фрагмента }}

Правила:
- chunk_id — ОБЯЗАТЕЛЬНО номер фрагмента, из которого взят ответ.
- Если информация есть — ты ОБЯЗАН выбрать один фрагмент.
- chunk_id может быть null, только если ответ "N/A".
- N/A разрешено ТОЛЬКО если ни один фрагмент не содержит ответа.
- Формат value должен строго соответствовать kind: {kind}.
- Никакого текста вне JSON.

Формат ответов:
number: Только цифры (пример: 122233 или 0.25). Без пробелов, букв, процентов.
name: Одно имя/наименование.
names: Несколько имён через запятую и пробел (Name One, Name Two).
boolean: Только true или false строчными.

Вопрос:
{question}

Контекст:
{context}
""")



def submit(filename, url, timeout=30):
    # отправляем на сервер
    with open(filename, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            timeout=timeout
        )

    print(f"Файл {filename} отправлен на {url}")
    print("HTTP status:", response.status_code)
    print("Response text:", response.text)

    return response


if __name__ == "__main__":
    EMAIL = "test@rag-tat.com"
    NAME = "Matashkov_v0"
    FILE_NAME = "submission_Matashkov_v0.json"
    URL = "http://5.35.3.130:800/submit"

    choice = input("1 – сгенерировать и отправить JSON; 2 – отправить JSON (выберите 1 или 2): ").strip()

    if choice == "1":
        # загрузка вопросов
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)

        answers = []

        for q in questions:
            print("Обрабатываю:", q["text"])
            ans = answer_question(q["text"], q["kind"])
            answers.append(ans)

        result = {
            "email": EMAIL,
            "submission_name": NAME,
            "answers": answers
        }
        with open(FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        submit(FILE_NAME, URL, 30)
    else:
        submit(FILE_NAME, URL, 30)



