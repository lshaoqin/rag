# Test timing for creating vectors on Pinecone
# For this, we will use the Simple Wikipedia dataset at https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.simple

import pyarrow.parquet as pq
import time
from embedding import create_embeddings_openai, create_embeddings_angle
from pinecone_utils import upsert_pinecone_index, query_pinecone_index
from milvus_utils import connect_to_milvus, create_milvus_collection, upsert_milvus, create_milvus_index, query_milvus
from pymilvus import FieldSchema, DataType

# OpenAI embeddings took 71.25210009992588 seconds, costs ~$0.08
def test_openai_embedding_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    wikipedia = wikipedia[['title', 'text']]

    start = time.perf_counter()
    embeddings = create_embeddings_openai(wikipedia['text'])
    end = time.perf_counter()

    print(f'OpenAI embeddings took {end - start} seconds')
    
    # save the embeddings to a file
    with open('embeddings.txt', 'w') as f:
        for item in embeddings:
            f.write("%s\n" % item)

# AnglE embeddings took 82.96513290004805 seconds (Non-GPU machine)
def test_angle_embedding_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    wikipedia = wikipedia[['title', 'text']]
    labelled = []
    for row in wikipedia['text']:
        labelled.append({'text': row[1]})

    start = time.perf_counter()
    embeddings = create_embeddings_angle(labelled)
    end = time.perf_counter()

    print(f'AnglE embeddings took {end - start} seconds')

# AnglE query embeddings took 5.134554299991578 seconds (Non-GPU machine)
# Seems like the model takes a while to load
def test_angle_query_embedding_time(query):
    start = time.perf_counter()
    embeddings = create_embeddings_angle([{'text': query}])
    end = time.perf_counter()
    print(f'AnglE query embeddings took {end - start} seconds')

# Pinecone upsert took 29.51726290001534 seconds
def test_pinecone_upsert_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    labels = wikipedia[['title']]

    for label in labels:
        # remove non-ascii characters
        labels.loc[:, label] = labels[label].str.encode('ascii', 'ignore').str.decode('ascii')

    embeddings = []
    with open('embeddings.txt', 'r') as f:
        for line in f:
            embeddings.append(line.strip())

    embeddings = [[float(value) for value in embedding[1:-2].split(", ")] for embedding in embeddings]

    # join the labels and embeddings
    items = []
    for i in range(len(labels)):
        items.append((labels.iloc[i]['title'], embeddings[i]))

    start = time.perf_counter()
    for i in range(10):
        response = upsert_pinecone_index(items[i*100:i*100+100], 'wikipedia', 'tom-rag')
        print(response)
    end = time.perf_counter()

    print(f'Pinecone upsert took {end - start} seconds')

'''
OpenAI embeddings took 0.8059008000418544 seconds
Pinecone query took 1.0907004999462515 seconds
Total time: 1.896601299988106 seconds

Question: What do I call the farming of seafood?
Results:
-----------
Aquaculture
Farm
Farming
Fish
Census of Marine Life
'''
def test_pinecone_query_time(query):

    start = time.perf_counter()
    embeddings = create_embeddings_openai(query)
    end = time.perf_counter()
    embedding_time = end - start
    print(f'OpenAI embeddings took {embedding_time} seconds')

    start = time.perf_counter()
    response = query_pinecone_index(embeddings[0], 'wikipedia', 'tom-rag')
    end = time.perf_counter()
    query_time = end - start

    print(f'Pinecone query took {query_time} seconds')
    print(f'Total time: {embedding_time + query_time} seconds')
    for item in response.matches:
        print(item.id)

def setup_milvus_for_AnglE():
    connect_to_milvus()
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    ]
    description = "Openai embeddings"
    create_milvus_collection("openai", fields, description)

def test_milvus_upsert_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    labels = wikipedia[['title']]

    for label in labels:
        # remove non-ascii characters
        labels.loc[:, label] = labels[label].str.encode('ascii', 'ignore').str.decode('ascii')

    embeddings = []
    with open('embeddings.txt', 'r') as f:
        for line in f:
            embeddings.append(line.strip())

    embeddings = [[float(value) for value in embedding[1:-2].split(", ")] for embedding in embeddings]

    # join the labels and embeddings
    items = []
    for i in range(len(labels)):
        items.append((labels.iloc[i]['title'], embeddings[i]))

    start = time.perf_counter()
    upsert_milvus(items, "openai")
    create_milvus_index("openai", "embedding", "IVF_FLAT", "L2")
    end = time.perf_counter()

    print(f'Milvus upsert took {end - start} seconds')

def test_milvus_query_time(query):
    embeddings = create_embeddings_openai([query])

    start = time.perf_counter()
    result = query_milvus(embeddings, "openai", 10, {"nprobe": 16})
    end = time.perf_counter()

    print(f'Milvus query took {end - start} seconds')
    print(result)

# test_openai_embedding_time()
# test_angle_query_embedding_time('What do I call the farming of seafood?')
# test_pinecone_upsert_time()
# test_pinecone_query_time(['What do I call the farming of seafood?'])
test_milvus_upsert_time()
test_milvus_query_time('What do I call the farming of seafood?')

