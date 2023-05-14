import os

os.environ['HF_HOME'] = 'e:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'e:/Large data/qa data/transformers/cache/'

import transformers
print(transformers.__version__)
import datasets
print(datasets.__version__)
from datasets import load_dataset, load_metric


datasets = load_dataset("squad_v2")
train_slice = datasets["train"]
print(train_slice[0]["context"])
print(train_slice[1]["context"])
passages = train_slice["context"]

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

questions = train_slice["question"]
# Calculate average number of sentences in each question
sentence_counts = [len(question.split('.')) for question in questions]
avg_sentences_per_question = sum(sentence_counts) / len(questions)
print("Average sentences per questions:", avg_sentences_per_question)
# Calculate average number of words in each question
word_counts = [len(sentence.split()) for question in questions for sentence in question.split('.')]
avg_words_per_sentence = sum(word_counts) / len(word_counts)
print("Average words per sentence in questions:", avg_words_per_sentence)



exit()
i=0
while True:
    # print(train_slice[i]["passage"])
    print(train_slice[i]["question"])
    i+=1
    print("Press Enter to continue or 'q' to quit...")
    user_input = input()
    if user_input.lower() == 'q':
        break

# calculate average length of passage
print("Continue...")
exit()