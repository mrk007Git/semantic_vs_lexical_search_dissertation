import os
import pickle
import numpy as np

emb_dir = 'models/embeddings'
files = sorted(os.listdir(emb_dir))

# Load two embeddings
with open(os.path.join(emb_dir, files[0]), 'rb') as f1, open(os.path.join(emb_dir, files[1]), 'rb') as f2:
    vec1 = pickle.load(f1)
    vec2 = pickle.load(f2)

# Print a sample and compare
print("Vec1[:5]:", vec1[:5])
print("Vec2[:5]:", vec2[:5])
print("Cosine similarity:", np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)))
