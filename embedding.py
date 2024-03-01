from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from angle_emb import AnglE, Prompts

load_dotenv()

def create_embeddings_openai(texts):
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def create_embeddings_angle(texts):
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    angle.set_prompt(prompt=Prompts.C)
    embeddings = angle.encode(texts, to_numpy=True)
    return embeddings