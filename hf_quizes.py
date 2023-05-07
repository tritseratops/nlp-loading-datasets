from transformers import pipeline

# ner = pipeline("ner", grouped_entities=True)
# print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))


classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")