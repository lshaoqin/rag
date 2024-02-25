# Connect to milvus server
# Credit to this tutorial by Stephen Collins for information on setting up milvus and text embedding
# https://dev.to/stephenc222/how-to-use-milvus-to-store-and-query-vector-embeddings-5hhl
from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_milvus_collection(name, fields, description):
    schema = CollectionSchema(fields, description)
    collection = Collection(name, schema, consistency_level="Strong")
    return collection

def drop_milvus_collection(name):
    collection = Collection(name)
    collection.drop()

def upsert_milvus(entities, name):
    collection = Collection(name)
    insert_result = collection.insert(entities)
    return insert_result

def create_milvus_index(name, field_name, index_type, metric_type):
    collection = Collection(name)
    index = collection.create_index(field_name, index_type, metric_type)
    return index

def query_milvus(query_entities, name, top_k, params):
    collection = Collection(name)
    query_result = collection.query(query_entities, top_k, params)
    return query_result