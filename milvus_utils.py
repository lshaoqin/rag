# Connect to milvus server
# Credit to this tutorial by Stephen Collins for information on setting up milvus and text embedding
# https://dev.to/stephenc222/how-to-use-milvus-to-store-and-query-vector-embeddings-5hhl
from pymilvus import connections, utility
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
    print(f"Collection {name} created.")
    return collection

def create_miniLM_collection(name):
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    ]
    description = "MiniLM embeddings"
    collection = Collection(name, CollectionSchema(fields, description), consistency_level="Strong")
    print(f"Collection {name} created.")
    return collection

def drop_milvus_collection(name):
    if utility.has_collection(name):
        collection = Collection(name)
        collection.drop()
        print(f"Collection {name} dropped.")
    else:
        print(f"Collection {name} does not exist.")

def upsert_milvus(entities, name):
    collection = Collection(name)
    insert_result = collection.insert(entities)
    print(f"Inserted {len(entities)} entities.")
    return insert_result

def create_milvus_index(collection_name, field_name, index_type, metric_type, params):
    collection = Collection(collection_name)
    index = {"index_type": index_type, "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index created for {collection_name}.")

def query_milvus(collection, search_vectors, search_field, search_params):
    result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["title"])
    return result[0]

def check_collection_exists(name):
    if utility.has_collection(name):
        print(f"Collection {name} exists.")
        return True
    else:
        print(f"Collection {name} does not exist.")
        return False

if __name__ == "__main__":
    connect_to_milvus()
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    ]
    description = "AnglE embeddings"
    create_milvus_collection("AnglE", fields, description)
    # upsert_milvus(embeddings, "openai")
    # create_milvus_index("openai", "embedding", "IVF_FLAT", "L2")
    # query_milvus(embeddings, "openai", 10, {"nprobe": 16})