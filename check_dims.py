from gensim.models import Word2Vec

try:
    cbow_model = Word2Vec.load("cbow_model.model")
    print(f"CBOW vector size: {cbow_model.vector_size}")
except:
    print("CBOW not found or error")

try:
    skipgram_model = Word2Vec.load("skipgram_model.model")
    print(f"Skipgram vector size: {skipgram_model.vector_size}")
except:
    print("Skipgram not found or error")
