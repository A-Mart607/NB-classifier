import json

with open('movie-review.NB', 'r') as f:
    model = json.load(f)

vocab = model['word_counts']

# Sort words by total count (pos + neg)
sorted_vocab = sorted(
    vocab.items(),
    key=lambda item: item[1]['pos'] + item[1]['neg'],
    reverse=True
)

top_freq = open('top_freq', "w", encoding='utf-8')

for word, counts in sorted_vocab:
    out = f"{word, counts['pos'] + counts['neg']}\n"
    top_freq.write(out)
    print(word, counts['pos'] + counts['neg'])
