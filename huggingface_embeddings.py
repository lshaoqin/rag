# Code from https://github.com/stephenc222/example-weaviate-vector-embeddings

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import os
import warnings
import numpy

# The transformers library internally is creating this warning, but does not
# impact our app. Safe to ignore.
warnings.filterwarnings(action='ignore', category=ResourceWarning)

# We won't have competing threads in this example app
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizer and model for UAE_Large
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # Alternative: WhereIsAI/UAE-Large-V1
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Move the model to the GPU if one is available
model = model.to(device)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def generate_embeddings_MiniLM_huggingface(text):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.numpy().tolist()[0]