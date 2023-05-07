# from here: https://visualstudiomagazine.com/Articles/2021/12/07/compute-ta-model-accuracy.aspx?Page=1
# imdb_hf_02_eval.py
# accuracy for tuned HF model for IMDB sentiment analysis
# Python 3.7.6  PyTorch 1.8.0  HF 4.11.3  Windows 10
# zipped raw data at:
# https://ai.stanford.edu/~amaas/data/sentiment/

import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizerFast
import torch as T
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import logging  # to suppress warnings

device = T.device('cpu')

class IMDbDataset(T.utils.data.Dataset):
  def __init__(self, reviews_lst, labels_lst):
    self.reviews_lst = reviews_lst  # list of token IDs
    self.labels_lst = labels_lst    # list of 0-1 ints

  def __getitem__(self, idx):
    item = {}  # [input_ids] [attention_mask] [labels]
    for key, val in self.reviews_lst.items():
      item[key] = T.tensor(val[idx]).to(device)
    item['labels'] = \
      T.tensor(self.labels_lst[idx]).to(device)
    return item

  def __len__(self):
    return len(self.labels_lst)

def read_imdb(root_dir):
  reviews_lst = []; labels_lst = []
  root_dir = Path(root_dir)
  for label_dir in ["pos", "neg"]:
    for f_handle in (root_dir/label_dir).iterdir():
      txt = f_handle.read_text(\
        encoding='utf-8')
      reviews_lst.append(txt)
      if label_dir == "pos":
        labels_lst.append(1)
      else:
        labels_lst.append(0)
  return (reviews_lst, labels_lst)  # lists of strings

def print_list(lst, front, back):
  # print first and last items
  n = len(lst)
  for i in range(front): print(lst[i] + " ", end="")
  print(" . . . ", end="")
  for i in range(back): print(lst[n-1-i] + " ", end="")
  print("")

def accuracy(model, ds, toker, num_reviews):
  # item-by-item: good for debugging but slow
  n_correct = 0; n_wrong = 0
  loader = DataLoader(ds, batch_size=1, shuffle=True)
  for (b_ix, batch) in enumerate(loader):
    print("====================================================")
    print(str(b_ix) + "  ", end="")
    input_ids = batch['input_ids'].to(device)  # just IDs, no masks

    # tensor([[101, 1045, 2253, . . 0, 0]])
    # words = toker.decode(input_ids[0])
    # [CLS] i went and saw . . [PAD] [PAD]

    lbl = batch['labels'].to(device)  # target 0 or 1
    mask = batch['attention_mask'].to(device)
    with T.no_grad():
      outputs = model(input_ids, \
        attention_mask=mask, labels=lbl)

    # SequenceClassifierOutput(
    #  loss=tensor(0.0168),
    #  logits=tensor([[-2.2251, 1.8527]]),
    #  hidden_states=None,
    #  attentions=None)
    logits = outputs[1]  # a tensor
    pred_class = T.argmax(logits)
    print("  target: " + str(lbl.item()), end="")
    print("  predicted: " + str(pred_class.item()), end="")
    if lbl.item() == pred_class.item():
      n_correct += 1; print(" | correct")
    else:
      n_wrong += 1; print(" | wrong")

    if b_ix == num_reviews - 1:
      break

    # if lbl.item() != pred_class.item():
    print("Test review as token IDs: ")
    T.set_printoptions(threshold=100, edgeitems=3)
    print(input_ids)
    print("Review source: ")
    words = toker.decode(input_ids[0])  # giant string
    print_list(words.split(' '), 3, 3)

  print("====================================================")

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  print("\nCorrect: %4d " % n_correct)
  print("Wrong:   %4d " % n_wrong)
  return acc

# wants 39GB RAM, crashes, OMG
def accuracy_fast(model, ds):
  # all items at once: slightly faster but less clear
  loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
  for (b_ix, batch) in enumerate(loader):  # one giant batch
    input_ids = batch['input_ids'].to(device)  # Size([200, 512])
    lbls = batch['labels'].to(device)  # all labels Size([200])

    mask = batch['attention_mask'].to(device)
    with T.no_grad():
      outputs = model(input_ids, \
        attention_mask=mask, labels=lbls)

    logits = outputs[1]  # all logits Size([200, 2])
    pred_y = T.argmax(logits, dim=1)  # 0s or 1s Size([200])

    num_correct = T.sum(lbls==pred_y)
    print("\nCorrect: ")
    print(num_correct.item())
    acc = (num_correct.item() * 1.0 / len(ds))
    return acc

def main():
  # 0. get ready
  print("\nBegin evaluation of IMDB HF model ")
  logging.set_verbosity_error()  # suppress wordy warnings
  T.manual_seed(1)
  np.random.seed(1)

  # 1. load pretrained model
  print("\nLoading (cached) untuned DistilBERT model ")
  model = \
    DistilBertForSequenceClassification.from_pretrained( \
    'distilbert-base-uncased')
  model.to(device)
  print("Done ")

  # 2. load tuned model wts and biases from file
  # print("\nLoading tuned model wts and biases ")
  # model.load_state_dict(T.load(".\\Models\\imdb_state.pt"))
  # model.eval()
  # print("Done ")

  # 3. load training data used to create tuned model
  print("\nLoading test data from file into memory ")
  # test_path = ".\\DataSmall\\aclImdb\\test"
  test_path = "e:\\Large data\\qa data\\20221124_imdb\\aclImdb\\test"
  test_texts, test_labels = read_imdb(test_path)
  print("0 in labels:", test_labels.count(0))
  print("1 in labels:", test_labels.count(1))
  print("Done ")

  # 4. tokenize the raw text data
  print("\nTokenizing test reviews data ")
  tokenizer = \
    DistilBertTokenizerFast.from_pretrained(\
    'distilbert-base-uncased')
  test_encodings = \
    tokenizer(test_texts, truncation=True, padding=True)
  print("Done ")

  # 5. put tokenized text into PyTorch Dataset
  print("\nConverting tokenized text into Pytorch Dataset ")
  test_dataset = IMDbDataset(test_encodings, test_labels)
  print("Done ")

  # 6. compute classification accuracy
  print("\nComputing model accuracy on first 5 test data ")
  acc = accuracy(model, test_dataset, tokenizer, num_reviews=100)
  print("Accuracy = %0.4f " % acc)

  # I need 39 GB for that
  # print("\nComputing model accuracy (fast) on test data ")
  # acc = accuracy_fast(model, test_dataset)
  # print("Accuracy = %0.4f " % acc)

  print("\nEnd demo ")

if __name__ == "__main__":
  main()
