from fastapi import FastAPI, File, UploadFile
from milvus_utils import connect_to_milvus, create_milvus_collection, upsert_milvus, generate_entities, create_milvus_index, query_milvus, check_collection_exists, drop_milvus_collection
from chunker import chunk_by_words
from huggingface_embeddings import generate_embeddings_MiniLM_huggingface
from pypdf import PdfReader

# To start, run uvicorn main:app --reload
app = FastAPI()
connect_to_milvus()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/{collection}/create")
async def create_collection(collection: str):
    try:
        create_milvus_collection(collection)
    except Exception as e:
        return {"error": str(e)}
    return {"result": "Collection created."}

@app.get("/{collection}/drop")
async def drop_collection(collection: str):
    try:
        result = drop_milvus_collection(collection)
    except Exception as e:
        return {"error": str(e)}
    return {"result": str(result)}

@app.post("/{collection}/upsert_txt")
async def upsert_txt(collection: str, file: UploadFile, chunk_size: int = 100, overlap_size: int = 50):
    try:
        contents = await file.read()
        contents = contents.decode("utf-8")
        chunks = chunk_by_words(contents, chunk_size, overlap_size)

        embeddings = []
        for chunk in chunks:
            embeddings.append(generate_embeddings_MiniLM_huggingface(chunk))

        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        entities = generate_entities(embeddings, chunks)
        result = upsert_milvus(entities, collection)

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(result)}

@app.post("/{collection}/upsert_pdf")
async def upsert_pdf(collection: str, file: UploadFile, chunk_size: int = 100, overlap_size: int = 50):
    try:
        reader = PdfReader(file.file)
        contents = ''
        for page in reader.pages:
            contents += page.extract_text()
        chunks = chunk_by_words(contents, chunk_size, overlap_size)

        embeddings = []
        for chunk in chunks:
            embeddings.append(generate_embeddings_MiniLM_huggingface(chunk))

        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        entities = generate_entities(embeddings, chunks)
        result = upsert_milvus(entities, collection)

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(result)}

@app.get("/{collection}/create_index")
async def create_index(collection: str):
    try:
        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        create_milvus_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

    except Exception as e:
        return {"error": str(e)}

    return {"result": "Index created."}

@app.post("/{collection}/query")
async def query(collection: str, text: str):
    try:
        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        embeddings = generate_embeddings_MiniLM_huggingface(text)
        result = query_milvus(collection, [embeddings], "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}})

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(result)}