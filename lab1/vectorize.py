import json as js
import pathlib as pl
import chromadb as cdb
import sentence_transformers as st

model = st.SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

folder = pl.Path(__file__).resolve().parent

database = cdb.PersistentClient(
    path=str(folder / "data" / "chroma_db"),
)

collection = database.get_or_create_collection(
    name="hr_e5_small",
    embedding_function=None,
    configuration={"hnsw": {"space": "cosine"}},
)

with (folder / "data" / "chunks.jsonl").open(
    encoding="utf-8"
) as file:

    for number, line in enumerate(file, start=1):
        chunk = js.loads(line)

        text = chunk["text"]
        model_input = "passage: " + text

        tokens = model.tokenizer(
            model_input,
            truncation=False,
        )["input_ids"]

        if len(tokens) > model.max_seq_length:
            raise ValueError(
                f"Чанк {number} слишком длинный: {len(tokens)} токенов"
            )

        vector = model.encode(
            model_input,
            normalize_embeddings=True,
        )

        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[vector.tolist()],
            documents=[text],
            metadatas=[chunk["metadata"]],
        )

        print("Сохранён чанк:", number)

print("Всего записей в БД:", collection.count())