from fastapi import FastAPI, File, UploadFile
from milvus_utils import connect_to_milvus, create_miniLM_collection, upsert_milvus, generate_entities, create_milvus_index, query_milvus, check_collection_exists
from chunker import chunk_by_words
from huggingface_embeddings import generate_embeddings_MiniLM_huggingface

app = FastAPI()
connect_to_milvus()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("{collection}/upsert_txt")
async def upsert_txt(collection: str, file: UploadFile, chunk_size: int = 100, overlap_size: int = 50):
    try:
        contents = await file.read()
        contents = contents.decode("utf-8")
        chunks = chunk_by_words(contents, chunk_size, overlap_size)

        embeddings = []
        for chunk in chunks:
            embeddings.append(generate_embeddings_MiniLM_huggingface(chunk))

        if not check_collection_exists(collection):
            create_miniLM_collection(collection)

        entities = generate_entities(embeddings, chunks)
        result = upsert_milvus(entities, collection)

    except Exception as e:
        return {"error": str(e)}

    return {"result": result}
