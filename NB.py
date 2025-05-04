import sys
import math
import json

"""
NB.py
should take the following parameters:
1. the training file,
2. the test file,
3. the file where the parameters of the resulting model will be saved,
4. and the output file where you will write predictions made by the classifier on the test data (one example per line).
    4.1 The last line in the output file should list the overall accuracy of the classifier on the test data.

The training and the test files should have the following format:
each line corresponds to an example;
first column is the label,
and the other columns are feature values.
"""

train_file = sys.argv[1]
test_file = sys.argv[2]

"""
example model:
{
  "vocab_size": 5000,
  "class_word_counts": {
    "pos": 123456,
    "neg": 117890
  },
  "word_counts": {
    "good": {"pos": 150, "neg": 20},
    "bad": {"pos": 10, "neg": 120},
    "fun": {"pos": 50, "neg": 10},
    ...
  },
  "class_doc_counts": {
    "pos": 12500,
    "neg": 12500
  }
}

"""
model_file = sys.argv[3]
output_file = sys.argv[4]

model = {
    'vocab_size' : 0,
    'class_word_counts' : {
        'pos' : 0,
        'neg' : 0
    },
    'class_doc_counts' : {
        'pos' : 0,
        'neg' : 0,
    },
    'word_counts' : dict(),
}
# step 1 process the training file and build the model

# don't want to assume that the entire vocab list is actually seen
seen_vocab = set()

with open(train_file, 'r', encoding="utf-8") as file:

    for vector in file:
        vector = vector.strip().split()
        _class = vector[0]

        # update Nc
        model['class_doc_counts'][_class] += 1
        for i in range(1, len(vector), 2):
            word = vector[i]
            count = int(vector[i + 1])

            # in case word hasn't been counted for vocab size yet
            seen_vocab.add(word)

            if word not in model['word_counts']:
                model['word_counts'][word] = {'pos': 0, 'neg': 0}

            # individual calculation for count (wi, c)
            model['word_counts'][word][_class] += count

            # helps for sigma w∈V count (w,c)
            model['class_word_counts'][_class] += count

model['vocab_size'] = len(seen_vocab)

with open(model_file, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)


pos_prior = model['class_doc_counts']['pos']
neg_prior = model['class_doc_counts']['neg']
total_doc_count = pos_prior + neg_prior

pos_prior /= total_doc_count
neg_prior /= total_doc_count

pos_prior = math.log(pos_prior)
neg_prior = math.log(neg_prior)

correct = 0
total = 0

output = open(output_file, "w", encoding='utf-8')

with open(test_file, 'r', encoding='utf-8') as file:

    for vector in file:
        total += 1

        vector = vector.strip().split()

        real_class = vector[0]

        pos_val = pos_prior
        neg_val = neg_prior

        # positive calculation
        for i in range(1, len(vector), 2):
            word = vector[i]
            count = int(vector[i + 1])

            # ignore words not seen in training as per page 6 of textbook
            if word not in seen_vocab:
                continue

            pos_num = model['word_counts'][word]['pos'] + 1
            pos_denom = model['class_word_counts']['pos'] + len(seen_vocab)
            log_pos = math.log(pos_num/pos_denom)

            pos_val += log_pos


            neg_num = model['word_counts'][word]['neg'] + 1
            neg_denom = model['class_word_counts']['neg'] + len(seen_vocab)
            log_neg = math.log(neg_num/neg_denom)

            neg_val += log_neg

        predicted_class = 'pos' if pos_val > neg_val else 'neg'

        output.write(f"{predicted_class}\n")
        if predicted_class == real_class:
            correct += 1

output.write(f"Overall Accuracy: {correct/total:.3f}")
print(f"Overall Accuracy: {correct/total:.3f}")
output.close()