import json


# read json and count stories and avg questions
filepath = "../../NLP/squad20/dev-v2.0.json"
# count yes/no questions based uif there is yes or no answer

# Opening JSON file
f = open(filepath)

# structure data/paragraphs/qas/  - whole doc/ story/ question and answer

# returns JSON object as
# a dictionary
json_data = json.load(f)
# item_dict `= json.loads(json_data)
print(len(json_data['data']))
#


# Iterating through the json
# list
# for i in data['emp_details']:
#     print(i)

# Closing file
f.close()


