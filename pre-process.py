import sys
import os
from string import punctuation

"""
Pre-processing: prior to building feature vectors,
- you should separate punctuation from words and lowercase the words in the reviews.

- You will train NB classifier on the training partition using the BOW features (use add-one smoothing).
- You will evaluate your classifier on the test partition.
- In addition to BOW features, you should experiment with additional features.
In that case, please provide a description of the features in your report.

Save the parameters of your BOW model in a file called movie-review-BOW.NB.
Report the accuracy of your program on the test data with BOW features.
"""
def break_text(text, vocab):
    negate_words = {'not', "didn't", "wasn't", "no", "never"}
    punctuation = [".", ",", "!", "?", ";", ":", "(", ")"]

    # hand selected from top freq list
    stopwords = [
        "the", "and", "a", "of", "to", "is", "in", "it", "this", "i",
        "was", "as", "with", "for", "on", "are", "you", "one"
    ]

    text = text.lower()
    tokens = text.split()
    output = dict()

    # negation feature
    negate = False
    for token in tokens:
        if token in vocab:
            if token in punctuation:
                negate = False

            if token in stopwords:
                negate = False
                continue

            if token in negate_words:
                negate = True
                output[token] = output.get(token, 0) + 1
                continue

            if negate:

                token = "NOT_" + token
                negate = False

            output[token] = output.get(token, 0) + 1

        else:
            # Separate basic punctuation from the token
            for punct in punctuation:
                token = token.replace(punct, f" {punct} ")
            new_tok = token.split()
            for new_token in new_tok:
                if new_token in punctuation:
                    negate = False


                if new_token in stopwords:
                    negate = False
                    continue

                if new_token in negate_words:
                    negate = True
                    output[new_token] = output.get(new_token, 0) + 1
                    continue

                if new_token in vocab:
                    if negate:
                        new_token = "NOT_" + new_token
                        negate = False

                    output[new_token] = output.get(new_token, 0) + 1
    return output



def make_vector(label, words):
    # words is a dict
    vector = [f"{word} {count}" for word,count in words.items()]
    return f"{label} {' '.join(vector)}"

# should take in something like 'train/'
input_directory = sys.argv[1].rstrip("/\\")

# every file directory should be train/neg + train/pos
neg_dir = os.listdir(f"{input_directory}/neg")

pos_dir = os.listdir(f"{input_directory}/pos")


out_name = os.path.basename(input_directory) + ".vectors"


output = open(f"{out_name}", "w", encoding='utf-8')

with open('imdb.vocab', 'r', encoding="utf-8") as f:
    # already in lowercase....but just in case
    vocab = set(line.strip().lower() for line in f)


for file in neg_dir:

    path = f"{input_directory}/neg/{file}"
    with open(path, 'r', encoding='utf-8') as f:

        text = f.read()
        words = break_text(text, vocab)
        vector = make_vector('neg', words)
        output.write(vector + '\n')


for file in pos_dir:
    path = f"{input_directory}/pos/{file}"
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        words = break_text(text, vocab)
        vector = make_vector('pos', words)
        output.write(vector + '\n')

output.close()
print("done processing", input_directory)