import json

with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    dicts = dict()
    dicts['ner'] = []
    dicts['relation'] = []
    dicts['event'] = []
    dicts['role'] = []
    for id in range(0, len(lines), 4):
        labels = lines[id + 2].strip().split('|')
        for label in labels:
            dicts['ner'].append(label.split(' ')[-1])
    dicts['ner'] = list(set(dicts['ner']))
    with open('valid_pattern.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(dicts, ensure_ascii=False, indent=4))