from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()

def create_embeddings_openai(texts):
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings
