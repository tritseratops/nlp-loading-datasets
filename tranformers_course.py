import os

os.environ['HF_HOME'] = 'e:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'e:/Large data/qa data/transformers/cache/'

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokens = tokenizer.tokenize("Let's try to tokenize!")
# print(tokens)
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(input_ids)
# final_input = tokenizer.prepare_for_model(input_ids)
# print(final_input)
# print(tokenizer.decode(input_ids))

# 2022.12.13
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# sequence = "I've been waiting for a HuggingFace course my whole life."
# sequence2 = "I don't know Greg, it sounds stupid"
#
# tokens1 = tokenizer.tokenize(sequence)
# ids1 = tokenizer.convert_tokens_to_ids(tokens1)
# tokens2 = tokenizer.tokenize(sequence2)
# ids2 = tokenizer.convert_tokens_to_ids(tokens2)
# print("Input ID 1:", ids1)
# print("Input ID 2:", ids2)
# # batched_ids =
# ids1 = [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
# ids2 = [1045, 2123, 1005, 1056, 2113, 6754, 1010, 2009, 4165, 5236, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id ]
# input_ids = torch.tensor([ids1, ids2])
# print("Input IDs:", input_ids)
#
# output = model(input_ids)
# print("Logits:", output.logits)
# print([tokenizer.decode(sentence_ids) for sentence_ids in input_ids])
# print(model(torch.tensor([ids1])).logits)
# print(model(torch.tensor([ids2])).logits)
# print(model(torch.tensor(input_ids)).logits)

# 2022.12.16
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [
#     [200, 200, 200],
#     [200, 200, tokenizer.pad_token_id],
# ]
# attention_mask = [
#     [1, 1, 1],
#     [1, 1, 0],
# ]
# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)).logits)

# 2022.12.18
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
#
# tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# input_ids = tokens['input_ids']
# print(input_ids)
# ids = input_ids.tolist()
# print(ids)
# print([tokenizer.decode(sentence_ids) for sentence_ids in ids])
# output = model(**tokens)

# 2022.12.19
# import torch
# from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
#
# # Same as before
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
#
# # This is new
# batch["labels"] = torch.tensor([1, 1])
#
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()

# 2022.12.20
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences2 = tokenizer(raw_datasets["train"]["sentence2"])
inputs = tokenizer("This is the first sentence", "This is the second one")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True
)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets =raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})