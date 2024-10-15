import json
with open('valid_pattern.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(set(data['ner'] + data['event'] +data['role'] + data['relation']))