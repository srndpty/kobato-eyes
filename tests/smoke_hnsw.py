import numpy as np

from index.hnsw import HNSWIndex

dim = 768
idx = HNSWIndex(space="cosine")
idx.build(dim, 128)
vecs = np.random.randn(10, dim).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
labels = list(range(1000, 1010))
idx.add(vecs, labels, num_threads=1)
print("OK:", idx.current_count)
