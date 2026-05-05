"""
Module 6 Week B — Core Skills Drill: Embedding Basics

Complete the three functions below to load, query, and compare
pre-trained GloVe word embeddings.
"""

import numpy as np

def load_glove(filepath):
    embeddings = {}
    count = 0 
    print ("#Task 1:\n")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue          
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
            
            count += 1 
            
            if count <= 5:
                print(f"--- Word #{count}: -> {word} ---")
                print(f"Vector Shape: {vector.shape}")
                print(f"First 3 values: {vector[:3]}") 
                print("-" * 30)
    print("*" * 50)            
    return embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. If either vector has zero norm, return 0.0.
    """
    # Calculate magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero-vector edge case
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Standard cosine similarity formula: (A . B) / (||A|| * ||B||)
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


def nearest_neighbors(word, embeddings, n=5):
    """Find the n most similar words to the given word.

    Returns a list of (word, score) tuples sorted by similarity descending,
    excluding the query word itself.
    """
    if word not in embeddings:
        return []
    #query_vec -> the vector representation of the input word  like "king" or "queen" etc.
    # query_vec equals -> number like [ 0.50451  0.68607 -0.59517 ......] for a 50-dimensional embedding
    query_vec = embeddings[word]
    results = []
    print (f"Finding nearest neighbors for '{word}'...\n")
    print(f"Query Vector Shape: {query_vec.shape}")
    print(f"First 3 values: {query_vec[:3]}\n")


    print("-" * 30)
    for target_word, target_vec in embeddings.items():
        #embeddings.items() -> returns a list of tuples -> (word(Name), vector(Array.NumberS)) like [("king", [0.50451, 0.68607, -0.59517, ...]), ("queen", [0.4321, 0.1234, -0.5678, ...]), ...]
        # Exclude the query word itself
        if target_word == word:
            continue
        
        sim = cosine_similarity(query_vec, target_vec)
        results.append((target_word, sim))

    # Sort by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:n]


if __name__ == "__main__":
    # Ensure the path matches your project structure
    glove = load_glove("data/glove_50k_50d.txt")
    
    if glove:
        print(f"Loaded {len(glove)} word vectors")

        # Task 2: Word similarity
        print("\n#Task 2:\n")
        sim = cosine_similarity(glove.get("king", np.zeros(50)), glove.get("queen", np.zeros(50)))
                               
        if sim is not None:
            print(f"cosine('king', 'queen') = {sim:.4f}")

        sim2 = cosine_similarity(glove.get("king", np.zeros(50)),
                                 glove.get("banana", np.zeros(50)))
        if sim2 is not None:
            print(f"cosine('king', 'banana') = {sim2:.4f}")
        print("*" * 50)
        # Task 3: Nearest neighbors
        print("\n#Task 3:\n")
        neighbors = nearest_neighbors("king", glove, n=5)
        if neighbors:
            print(f"{'Rank':<5} | {'Word':<12} | {'Similarity':<10}")
            print("-" * 35)
            for i, (word, sim) in enumerate(neighbors, 1):
                print(f"{i:<5} | {word:<12} | {sim:<10.4f}")
