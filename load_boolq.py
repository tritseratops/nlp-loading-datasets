import os

os.environ['HF_HOME'] = 'e:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'e:/Large data/qa data/transformers/cache/'

import transformers
print(transformers.__version__)
import datasets
print(datasets.__version__)
from datasets import load_dataset, load_metric


datasets = load_dataset("boolq")
train_slice = datasets["train"]
print(train_slice[0]["passage"])
print(train_slice[1]["passage"])
passages = train_slice["passage"]

lengths = list(map(len, passages))
average_length = sum(lengths) / len(passages)

print("Average length of passages:", average_length)

# Calculate average number of sentences in each passage
sentence_counts = [len(passage.split('.')) for passage in passages]
avg_sentences_per_passage = sum(sentence_counts) / len(passages)
print("Average sentences per passage:", avg_sentences_per_passage)
# Calculate average number of words in each sentence
word_counts = [len(sentence.split()) for passage in passages for sentence in passage.split('.')]
avg_words_per_sentence = sum(word_counts) / len(word_counts)
print("Average words per sentence:", avg_words_per_sentence)
# calculate average length of passage
print("Continue...")
exit()