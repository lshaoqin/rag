from fastapi import FastAPI, File, UploadFile
from milvus_utils import connect_to_milvus, create_milvus_collection, upsert_milvus, create_milvus_index, query_milvus

app = FastAPI()
connect_to_milvus()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upsert_txt")
async def upsert_txt(file: UploadFile):
    contents = await file.read()
    contents = contents.decode("utf-8")
    
    return {"filename": file.filename, "contents": contents}
