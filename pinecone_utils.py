from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone, PodSpec
import os

load_dotenv()

def create_chunks(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs

def create_pinecone_index(name):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    pc.create_index(
    name="tom-rag",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment='us-west-2',
        pod_type='p1.x1'
    )
)

def upsert_pinecone_index(vectors, namespace, name):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(name)

    upsert_response = index.upsert(vectors=vectors, namespace=namespace)

    return upsert_response

def query_pinecone_index(query_vector, namespace, name, top_k=5):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(name)

    query_response = index.query(vector=query_vector, namespace=namespace, top_k=top_k, include_metadata=True, include_values=True)

    return query_response