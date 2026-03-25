from gensim.models import Word2Vec
import numpy as np
import os

# Set current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    cbow_model = Word2Vec.load("cbow_model.model")
    word = "research"
    if word in cbow_model.wv:
        vector = cbow_model.wv[word]
        # Format as requested: "Word - val1, val2, ..."
        vector_str = ", ".join([f"{x:.4f}" for x in vector])
        print(f"{word} - {vector_str}")
    else:
        # Fallback to another word if 'research' is missing
        keys = list(cbow_model.wv.key_to_index.keys())
        if keys:
            word = keys[0]
            vector = cbow_model.wv[word]
            vector_str = ", ".join([f"{x:.4f}" for x in vector])
            print(f"{word} - {vector_str}")
        else:
            print("Vocabulary empty")

except Exception as e:
    print(f"Error: {e}")
