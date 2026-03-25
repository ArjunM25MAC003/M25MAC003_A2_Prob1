"""m25mac003_prob1
# **PROBLEM 1: LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA**

Objective: The objective of this assignment is to train 
           Word2Vec models on textual data collected from IIT Jodhpur 
           sources and analyze the semantic structure 
           captured by the learned embeddings.
This script performs the following tasks:
1. Scrapes textual data from IIT Jodhpur websites
2. Extracts text from PDF documents
3. Cleans and preprocesses the corpus
4. Trains CBOW and Skip-gram Word2Vec models
5. Evaluates semantic similarity
6. Visualizes embeddings using t-SNE 
"""


# TASK-1: DATASET PREPARATION

# Imports
import os
import re
import requests
import pdfplumber
import nltk
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from collections import Counter

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Web scraping from IIT Jodhpur websites and faculty pages
website_urls = [
    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/mathematics/",
]

faculty_urls = [
    "https://anandmishra22.github.io/",
]

scraped_documents = []

def fn_scrape_page(url):
    # Extracts paragraph text from given webpages
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    text = ""
    for p in soup.find_all("p"):
        text += p.get_text() + " "
    return text

# Scrape website pages
for url in website_urls:
    print("Scraping:", url)
    scraped_documents.append(fn_scrape_page(url))

# Scrape faculty pages
for url in faculty_urls:
    print("Scraping:", url)
    scraped_documents.append(fn_scrape_page(url))

print("Total scraped documents:", len(scraped_documents))

# Using Academic Regulation PDFs from IIT Jodhpur website for text extraction
print("Now upload Academic Regulation PDFs")

def fn_extract_pdf(pdf_path):
    # EXtracts all text from a pdf file
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
    return full_text

pdf_folder = "pdf" # Folder containing the required documents  
pdf_documents = []

for filename in os.listdir(pdf_folder):
    filepath = os.path.join(pdf_folder, filename)
    print("Processing:", filepath)

    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

    pdf_documents.append(full_text)

print("Total PDF documents:", len(pdf_documents))

# Text Preprocessing and Tokenization
all_raw_documents = scraped_documents + pdf_documents
documents = []

for doc in all_raw_documents:
    doc = re.sub(r'\d+/\d+/\d+', '', doc)   # remove dates
    doc = re.sub(r'\d+:\d+\s*(AM|PM)', '', doc)   # remove times
    doc = re.sub(r'\s+', ' ', doc)      #remove extra spaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)   # keep only alphabets and spaces
    doc = doc.lower()             # convert to lowercase

    tokens = word_tokenize(doc)    # tokenization
    tokens = [w for w in tokens if len(w) > 2]    # removing very short words

    documents.append(tokens)

print("Total combined documents:", len(documents))

# CORPUS STATISTICS
total_docs = len(documents)
total_tokens = sum(len(doc) for doc in documents)
vocab = set(word for doc in documents for word in doc)

print("Total Documents:", total_docs)
print("Total Tokens:", total_tokens)
print("Vocabulary Size:", len(vocab))

with open("clean_corpus.txt", "w") as f:
    for doc in documents:
        f.write(" ".join(doc) + "\n")

print("Clean corpus saved as clean_corpus.txt")

# STOPWORD REMOVAL 
from gensim.models import Word2Vec
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Add domain-specific stopwords
custom_stopwords = {
    "shall", "may", "institute", "indian",
    "technology", "jodhpur", "iit", "also"
}

all_stopwords = stop_words.union(custom_stopwords)

documents = []

with open("clean_corpus.txt", "r") as f:
    for line in f:
        tokens = line.strip().split()
        tokens = [w for w in tokens if w not in all_stopwords]
        documents.append(tokens)

print("Reloaded with extended stopword removal.")


# TASK-2 : MODEL TRAINING

# CBOW MODEL
cbow_model = Word2Vec(
    documents,
    vector_size=200,    # dimensions of word vectors
    window=8,           # context window size
    min_count=1,       # include rare words
    sg=0,              # CBOW
    negative=10        # negative sampling
)

skipgram_model = Word2Vec(
    sentences=documents,
    vector_size=100,   # dimensions of word vectors
    window=5,        # context window size
    min_count=2,     # include rare words
    workers=4,      # parallelization
    sg=1,        # Skip-gram
    negative=5
)

print("Skip-gram model trained successfully.")

# Saving models
cbow_model.save("cbow_model.model")
skipgram_model.save("skipgram_model.model")

print("CBOW - research:", cbow_model.wv.most_similar("research", topn=5))
print("CBOW - student:", cbow_model.wv.most_similar("student", topn=5))
print("CBOW - phd:", cbow_model.wv.most_similar("phd", topn=5))
print("CBOW - exam:", cbow_model.wv.most_similar("exam", topn=5))

print("Skip-gram - research:", skipgram_model.wv.most_similar("research", topn=5))

"""analogy

"""

# TASK-3: SEMENTIC ANALYSIS
from collections import Counter

all_tokens = [word for doc in documents for word in doc]
freq = Counter(all_tokens)
print(freq.most_common(30))


# Word cloud visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of IIT Jodhpur Corpus", fontsize=20)
plt.show()  

test_words = ["student", "course", "semester", "program", "research", "registration", "grade"]

# 1. top 5 nearest neighbors for each test word
for word in test_words:
    if word in cbow_model.wv:
        print(f"\nCBOW - {word}")
        print(cbow_model.wv.most_similar(word, topn=5))


# 2. ANALOGY TASK
cbow_model.wv.most_similar(
    positive=["course", "grade"],
    negative=["student"],
    topn=5
)

cbow_model.wv.most_similar(
    positive=["course", "registration"],
    negative=["semester"],
    topn=5
)

cbow_model.wv.most_similar(
    positive=["course", "project"],
    negative=["research"],
    topn=5
)


# TASK-4: VISUALIZATION (PCA / t-SNE)"""

words = [
    "student", "students", "course", "courses",
    "semester", "program", "academic",
    "registration", "grade", "research",
    "project", "faculty", "exam"
]

words = [w for w in words if w in cbow_model.wv.key_to_index]
print("Words used:", words)

# CBOW t-SNE 
vectors = [cbow_model.wv[w] for w in words]
vectors_array = np.array(vectors) # Convert list of vectors to a NumPy array

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
result = tsne.fit_transform(vectors_array)

plt.figure(figsize=(8,6))
for i, word in enumerate(words):
    plt.scatter(result[i,0], result[i,1])
    plt.text(result[i,0]+0.5, result[i,1]+0.5, word)

plt.title("CBOW Word Embeddings (t-SNE)")
plt.show()

skipgram_model = Word2Vec.load("skipgram_model.model")

words = [
    "student", "students", "course", "courses",
    "semester", "program", "academic",
    "registration", "grade", "research",
    "project", "faculty", "exam"
]
words = [w for w in words if w in skipgram_model.wv.key_to_index]

vectors_skip = [skipgram_model.wv[w] for w in words]
vectors_skip_array = np.array(vectors_skip) # Convert list of vectors to a NumPy array

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
result_skip = tsne.fit_transform(vectors_skip_array)

plt.figure(figsize=(8,6))
for i, word in enumerate(words):
    plt.scatter(result_skip[i,0], result_skip[i,1])
    plt.text(result_skip[i,0]+0.5, result_skip[i,1]+0.5, word)

plt.title("Skip-gram Word Embeddings (t-SNE)")
plt.show()