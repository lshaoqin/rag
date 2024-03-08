# Test timing for creating vectors on Pinecone
# For this, we will use the Simple Wikipedia dataset at https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.simple

import pyarrow.parquet as pq
import numpy as np
import time
from embedding import create_embeddings_openai, create_embeddings_angle
from pinecone_utils import upsert_pinecone_index, query_pinecone_index
from milvus_utils import connect_to_milvus, create_milvus_collection, upsert_milvus, create_milvus_index, query_milvus, drop_milvus_collection
from pymilvus import FieldSchema, DataType, Collection
from huggingface_embeddings import generate_embeddings_UAE_huggingface
import torch

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
        labelled.append({'text': row})

    start = time.perf_counter()
    embeddings = create_embeddings_angle(labelled)
    end = time.perf_counter()

    torch.save(embeddings, 'embeddings_angle.pt')

    print(f'AnglE embeddings took {end - start} seconds')

def test_UAE_embedding_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    wikipedia = wikipedia[['title', 'text']]

    start = time.perf_counter()
    embeddings = []
    for row in wikipedia['text']:
        embeddings.append(generate_embeddings_UAE_huggingface(row))
    end = time.perf_counter()

    print(f'UAE embeddings took {end - start} seconds')

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
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    ]
    description = "AnglE embeddings"

    drop_milvus_collection("angle")
    create_milvus_collection("angle", fields, description)

# Milvus upsert took 1.2594313000008697 seconds
def test_milvus_upsert_time():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    labels = wikipedia[['title']]

    for label in labels:
        # remove non-ascii characters
        labels.loc[:, label] = labels[label].str.encode('ascii', 'ignore').str.decode('ascii')

    embeddings = torch.load('embeddings_angle.pt')

    # join the labels and embeddings
    entries = [
        [str(i) for i in range(len(labels))],
        embeddings.tolist(),
        labels['title'].tolist()
    ]

    connect_to_milvus()

    start = time.perf_counter()
    upsert_milvus(entries, "angle")
    create_milvus_index("angle", "embeddings", "IVF_FLAT", "L2", {"nlist": 128})
    end = time.perf_counter()

    print(f'Milvus upsert took {end - start} seconds')

'''
Milvus query took 0.24138000000675675 seconds
["id: 916, distance: 203.069091796875, entity: {'title': 'Million'}", 
"id: 3, distance: 203.3989715576172, entity: {'title': 'A'}"]
'''
def test_milvus_query_time(query):    
    collection = Collection("UAE_huggingface")
    collection.load()

    start = time.perf_counter()
    query_embeddings = generate_embeddings_UAE_huggingface(query)
    end = time.perf_counter()

    print(f'UAE query embeddings took {end - start} seconds')

    start = time.perf_counter()
    result = query_milvus(collection, [query_embeddings], "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}})
    end = time.perf_counter()

    print(f'Milvus query took {end - start} seconds')
    print(result)
"""
UAE embeddings took 304.0680951999966 seconds
Milvus upsert took 2.5230127000104403 seconds
UAE query embeddings took 0.4172201000037603 seconds
Milvus query took 0.41844780000974424 seconds
["id: 419, distance: 0.9305093288421631, entity: {'title': 'United States customary units'}", 
"id: 676, distance: 0.9305093288421631, entity: {'title': 'Telford United F.C.'}", 
"id: 38, distance: 0.9595455527305603, entity: {'title': 'Boot device'}"]
"""
def test_UAE_with_milvus(query):
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    labels = wikipedia[['title']]

    start = time.perf_counter()
    embeddings = []
    for row in wikipedia['text']:
        embeddings.append(generate_embeddings_UAE_huggingface(row))
    end = time.perf_counter()

    print(f'UAE embeddings took {end - start} seconds')

    for label in labels:
        # remove non-ascii characters
        labels.loc[:, label] = labels[label].str.encode('ascii', 'ignore').str.decode('ascii')

    connect_to_milvus()

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    ]
    description = "UAE_huggingface embeddings"
    drop_milvus_collection("UAE_huggingface")
    create_milvus_collection("UAE_huggingface", fields, description)

    entries = [
        [str(i) for i in range(len(labels))],
        embeddings,
        labels['title'].tolist()
    ]

    start = time.perf_counter()
    upsert_milvus(entries, "UAE_huggingface")
    create_milvus_index("UAE_huggingface", "embeddings", "IVF_FLAT", "L2", {"nlist": 128})
    end = time.perf_counter()

    print(f'Milvus upsert took {end - start} seconds')
    
    collection = Collection("UAE_huggingface")
    collection.load()

    start = time.perf_counter()
    query_embeddings = generate_embeddings_UAE_huggingface(query)
    end = time.perf_counter()

    print(f'UAE query embeddings took {end - start} seconds')

    start = time.perf_counter()
    result = query_milvus(collection, [query_embeddings], "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}})
    end = time.perf_counter()

    print(f'Milvus query took {end - start} seconds')
    print(result)

def titles_file():
    wikipedia = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    wikipedia = wikipedia[:1000]
    labels = wikipedia[['title']]
    labels.to_csv('titles.csv', index=False)

# test_openai_embedding_time()
# test_angle_query_embedding_time('What do I call the farming of seafood?')
test_angle_embedding_time()
# test_pinecone_upsert_time()
# test_pinecone_query_time(['What do I call the farming of seafood?'])
# setup_milvus_for_AnglE()
# test_milvus_upsert_time()
# test_milvus_query_time('What do I call the farming of seafood?')
# test_UAE_embedding_time()
# test_UAE_with_milvus('What do I call the farming of seafood?')
# titles_file()