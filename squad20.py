import json


# read json and count stories and avg questions
filepath_dev = "../../NLP/squad20/dev-v2.0.json"
filepath_train = "../../NLP/squad20/train-v2.0.json"
# count yes/no questions based uif there is yes or no answer

# Opening JSON file
# f = open(filepath_dev)
f = open(filepath_train)

# structure data/paragraphs/qas/  - whole doc/ story/ question and answer

# returns JSON object as
# a dictionary
json_data = json.load(f)
# item_dict `= json.loads(json_data)
#


# Iterating through the json
# list

total_fields = json_data['data']
total_questions = 0
total_paragraphs = 0
for data_fields in total_fields:
    print(data_fields["title"])
    paragraphs = len(data_fields)
    print("Paragraphs:", paragraphs)
    total_paragraphs += paragraphs
    data_field_questions = 0
    for paragraph in data_fields["paragraphs"]:
        p_questions = len(paragraph["qas"])
        data_field_questions += p_questions
    print("Data Field questions: ", data_field_questions)
    total_questions += data_field_questions

print("Fields:", len(total_fields))
print("Total Paragraphs: ", total_paragraphs)
print("Total questions: ", total_questions)

# Closing file
f.close()


