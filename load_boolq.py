import os

os.environ['HF_HOME'] = 'e:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'e:/Large data/qa data/transformers/cache/'

import transformers
print(transformers.__version__)
import datasets
print(datasets.__version__)
from datasets import load_dataset, load_metric


datasets = load_dataset("boolq")

print("Continue...")
exit()