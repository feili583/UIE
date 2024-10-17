import json
import os

def find_str(text, substring):
    words = text.split(' ')
    substring_words = substring.split(' ')

    for i in range(len(words) - len(substring_words) + 1):
        if words[i:i + len(substring_words)] == substring_words:
            start_index = i
            return start_index

def coll2pot(path):
    valid_pattern = dict()
    sentences = []
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            valid_pattern['ner'] = [ty.replace(' ', '-') for ty in line['schema']['ent']]
            valid_pattern['event'] = [ty.replace(' ', '-') for ty in list(line['schema']['event'].keys())]
            # print(line['schema']['event'].values())
            valid_pattern['role'] = [ty.replace(' ', '-') for ty in list(set([item for sublist in line['schema']['event'].values() for item in sublist]))]
            ners = []
            events = []
            roles = []
            sentence = line['text']
            sentences.append(sentence)
            for ner in line['ans']['ent']:
                start = find_str(sentence, ner['text'])
                end = start + len(ner['text'].split(' ')) - 1
                ners.append([start, end, ner['type'].replace(' ', '-')])
            for event in line['ans']['event']:
                trigger_start = find_str(sentence, event['trigger']['text'])
                trigger_end = trigger_start + len(event['trigger']['text'].split(' ')) - 1
                events.append([trigger_start, trigger_end, event['event_type'].replace(' ', '-')])
                for role in event['args']:
                    start = find_str(sentence, role['text'])
                    end =  start + len(role['text'].split(' ')) - 1
                    roles.append([trigger_start, trigger_end, start, end, role['role'].replace(' ', '-')])
            tags.append(ners + events + roles)

    return sentences, tags, valid_pattern

def pot2file(path, file, sentences, tags, valid_pattern, valid_pattern_path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + file, 'w', encoding='utf-8') as f:
        for sentence, tag in zip(sentences, tags):
            f.write(sentence + '\n')
            tmp = []
            for span in tag:
                if len(span) == 3:
                    tmp.append(str(span[0]) + ',' + str(span[1] + 1) + ' ' + span[2])
                if len(span) == 5:
                    if span[0] < span[2]:
                        for start in range(span[0], span[1] + 1):
                            for end in range(span[2], span[3] + 1):
                                tmp.append(str(start) + ',' + str(end + 1) + ' ' + span[-1])
                    else:
                        for start in range(span[2], span[3] + 1):
                            for end in range(span[0], span[1] + 1):
                                tmp.append(str(start) + ',' + str(end + 1) + ' r_' + span[-1])
                                if 'r_' + span[-1] not in valid_pattern['role']:
                                    valid_pattern['role'].append('r_' + span[-1])
            f.write('|'.join(tmp))
            f.write('\n')
            tmp = []
            for span in tag:
                if len(span) == 3:
                    tmp.append(str(span[0]) + ',' + str(span[1] + 1) + ' ' + span[2])
                if len(span) == 5:
                    if span[0] < span[2]:
                        tmp.append(str(span[0]) + ',' + str(span[-2] + 1) + ' ' + span[-1])
                    else:
                        tmp.append(str(span[2]) + ',' + str(span[1] + 1) + ' r_' + span[-1])
            f.write('|'.join(tmp))
            f.write('\n\n')
    if 'train' in file:
        with open(valid_pattern_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(valid_pattern, ensure_ascii=False, indent=4))
        print('all tags', valid_pattern['ner'] + valid_pattern['role'] + valid_pattern['event'])
        print(len(valid_pattern['ner']), len(valid_pattern['role']), len(valid_pattern['event']))
                

if __name__ == '__main__':
    print('casie')
    
    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/train.jsonl'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    print('train sentences', len(sentences))
    pot2file(save_path, 'train.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    
    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/test.jsonl'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    print('test sentences', len(sentences))
    pot2file(save_path, 'test.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)

    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/dev.jsonl'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/event/casie/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    print('dev sentences', len(sentences))
    pot2file(save_path, 'dev.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)