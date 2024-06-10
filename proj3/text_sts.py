# STEP 1
# sts; sentence text similarity  # pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

# STEP 2
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3
sentences1 = "The weather is lovely today."
sentences2 = "It's so sunny outside!"

# STEP 4
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

print(embeddings1.shape)
print(embeddings2.shape)

# STEP 5
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)