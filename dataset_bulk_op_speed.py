from datasets import load_dataset
import time

dataset = load_dataset("boolq")
passages = dataset["train"]["passage"]

start_time = time.time()

lengths = [len(passage) for passage in passages]
average_length = sum(lengths) / len(lengths)

end_time = time.time()

print("Average length of passages:", average_length)
print("Execution time:", end_time - start_time)

"""
Output:

Average length of passages: 565.6130264134931
Execution time: 0.00099945068359375
"""
"""
Version using `map` and `sum`:
"""
from datasets import load_dataset
import time

dataset = load_dataset("boolq")
passages = dataset["train"]["passage"]

start_time = time.time()

lengths = dataset["train"].map(lambda x: {"passage_length" : len(x["passage"])})
average_length = sum(lengths["passage_length"]) / len(lengths)

end_time = time.time()

print("Average length of passages:", average_length)
print("Execution time:", end_time - start_time)

"""
Average length of passages: 565.6130264134931
Execution time: 0.2606818675994873
"""