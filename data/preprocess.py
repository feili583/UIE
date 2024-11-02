import json
import os

paths = ['/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/absa/14lap/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/absa/14res/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/absa/15res/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/absa/16res/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/pot',  \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/cadec/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/conll2003', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/EE/ACE', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/EE/ACE_add', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/genia', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/rel/conll04/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/rel/nyt/pot', \
                '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/rel/scierc/pot'
                ]
save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/all/'
all_data = []
all_cross_labels = []
all_labels = []
valid_pattern = dict()
valid_pattern['ner'] = []
valid_pattern['relation'] = []
valid_pattern['event'] = []
valid_pattern['role'] = []

for path in paths:
    if os.path.exists(path+'/train.txt'):
        with open(path+'/train.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    elif os.path.exists(path+'/pot_train_1224.txt'):
        with open(path+'/pot_train_1224.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    else:
        with open(path+'/train.data', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    with open(path+'/valid_pattern.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        valid_pattern['ner'] += data['ner']
        valid_pattern['relation'] += data['relation']
        valid_pattern['event'] += data['event']
        valid_pattern['role'] += data['role']
with open(save_path+'train.txt', 'w', encoding='utf-8') as f:
    for id in range(len(all_data)):
        f.write(all_data[id])
        f.write(all_cross_labels[id])
        f.write(all_labels[id]+'\n')
print(len(all_data))
with open(save_path+'valid_pattern.json', 'w', encoding='utf-8') as f:
    all_tags = []
    valid_pattern['ner'] = list(set(valid_pattern['ner']))
    valid_pattern['relation'] = list(set(valid_pattern['relation']))
    valid_pattern['event'] = list(set(valid_pattern['event']))
    valid_pattern['role'] = list(set(valid_pattern['role']))
    all_tags += (valid_pattern['ner'] + valid_pattern['relation'] + valid_pattern['event'] + valid_pattern['role'])
    f.write(json.dumps(valid_pattern,ensure_ascii=False,indent=4))
    print(all_tags)
    print(len(all_tags))
    print('ner', len(valid_pattern['ner']))
    print('relation', len(valid_pattern['relation']))
    print('event', len(valid_pattern['event']))
    print('role', len(valid_pattern['role']))

all_data = []
all_cross_labels = []
all_labels = []
for path in paths:
    if os.path.exists(path+'/dev.txt'):
        with open(path+'/dev.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    elif os.path.exists(path+'/pot_dev_1224.txt'):
        with open(path+'/pot_dev_1224.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    else:
        with open(path+'/dev.data', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
print(len(all_data))
with open(save_path+'dev.txt', 'w', encoding='utf-8') as f:
    for id in range(len(all_data)):
        f.write(all_data[id])
        f.write(all_cross_labels[id])
        f.write(all_labels[id]+'\n')

all_data = []
all_cross_labels = []
all_labels = []
for path in paths:
    if os.path.exists(path+'/test.txt'):
        with open(path+'/test.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    elif os.path.exists(path+'/pot_test_1224.txt'):
        with open(path+'/pot_test_1224.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
    else:
        with open(path+'/test.data', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for id in range(0, len(lines), 4):
                all_data.append(lines[id])
                all_cross_labels.append(lines[id+1])
                all_labels.append(lines[id+2])
print(len(all_data))
with open(save_path+'test.txt', 'w', encoding='utf-8') as f:
    for id in range(len(all_data)):
        f.write(all_data[id])
        f.write(all_cross_labels[id])
        f.write(all_labels[id]+'\n')