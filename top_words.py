import collections

filename = 'clean_corpus.txt'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower().split()
except FileNotFoundError:
    print("clean_corpus.txt not found.")
    exit(1)

counts = collections.Counter(text)
top_10 = counts.most_common(10)

# Format: word1, frequence, word2,frequencey, ...
output_parts = []
for word, freq in top_10:
    output_parts.append(f"{word}, {freq}")

print(", ".join(output_parts))
