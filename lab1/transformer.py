import sentence_transformers as st

# Загружаем готовую модель. При первом запуске она скачивается.
model = st.SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

text = "Основной отпуск составляет 28 календарных дней."

# Вычисляем вектор одного текста.
vector = model.encode(
    "passage: " + text,
    normalize_embeddings=True,
)

print("Количество чисел:", len(vector))
print("Первые пять чисел:", vector[:5])

documents = [
    "Основной отпуск составляет 28 календарных дней.",
    "Компенсация интернета составляет 1 200 рублей.",
]

question = "Сколько дней я могу отдыхать за год?"

question_vector = model.encode(
    "query: " + question,
    normalize_embeddings=True,
)

for document in documents:
    document_vector = model.encode(
        "passage: " + document,
        normalize_embeddings=True,
    )

    similarity = float(question_vector @ document_vector)

    print("Документ:", document)
    print("Сходство:", similarity)
    print()