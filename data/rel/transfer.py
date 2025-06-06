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
    sentences = []
    tags = []
    valid_pattern = dict()
    valid_pattern['ner'] = []
    valid_pattern['relation'] = []
    valid_pattern['event'] = []
    valid_pattern['role'] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            ners = []
            relations = []
            line = json.loads(line)
            sentence = line['text']
            sentences.append(sentence)
            valid_pattern['ner'] = [tag.replace(' ', '-') for tag in line['schema']['ent']]
            valid_pattern['relation'] = [tag.replace(' ', '-') for tag in line['schema']['rel']]
            for ner in line['ans']['ent']:
                start = find_str(sentence, ner['text'])
                end = start + len(ner['text'].split(' ')) - 1
                ners.append([start, end, ner['type'].replace(' ','-')])
            for rel in line['ans']['rel']:
                ner1 = rel['head']['text']
                start1 = find_str(sentence, ner1)
                end1 = start1 + len(ner1.split(' ')) - 1
                ner2 = rel['tail']['text']
                start2 = find_str(sentence, ner2)
                end2 = start2 + len(ner2.split(' ')) - 1
                relations.append([start1, end1, start2, end2, rel['relation'].replace(' ','-')])
            tags.append(ners + relations)

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
                    if span[0] < span[2] or (span[0] == span[2] and span[1] < span[3]):
                        for start in range(span[0], span[1] + 1):
                            for end in range(span[2], span[3] + 1):
                                tmp.append(str(min(start, end)) + ',' + str(max(start, end) + 1) + ' ' + span[-1])
                    else:
                        for start in range(span[2], span[3] + 1):
                            for end in range(span[0], span[1] + 1):
                                tmp.append(str(min(start, end)) + ',' + str(max(start, end) + 1) + ' r_' + span[-1])
                                if 'r_' + span[-1] not in valid_pattern['relation']:
                                    valid_pattern['relation'].append('r_' + span[-1])
            f.write('|'.join(tmp))
            f.write('\n')
            tmp = []
            for span in tag:
                if len(span) == 3:
                    tmp.append(str(span[0]) + ',' + str(span[1] + 1) + ' ' + span[2])
                if len(span) == 5:
                    if span[0] < span[2] or (span[0] == span[2] and span[1] < span[3]):
                        tmp.append(str(min(span[0], span[2])) + ',' + str(max(span[-2], span[1]) + 1) + ' ' + span[-1])
                    else:
                        tmp.append(str(min(span[0], span[2])) + ',' + str(max(span[-2], span[1]) + 1) + ' r_' + span[-1])
            f.write('|'.join(tmp))
            f.write('\n\n')
    if 'train' in file:
        with open(valid_pattern_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(valid_pattern, ensure_ascii=False, indent=4))
        print('all tags', valid_pattern['ner'] + valid_pattern['relation'])
        print(len(valid_pattern['ner']), len(valid_pattern['relation']))

if __name__ == '__main__':
    print('conll04')
    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/train.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    
    pot2file(save_path, 'train.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('train sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/test.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'test.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('test sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/dev.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/conll04/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'dev.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('dev sentences', len(sentences))

    print('nyt')
    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/train.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    # print('all tags', valid_pattern['ner'] + valid_pattern['relation'])
    pot2file(save_path, 'train.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('train sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/test.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'test.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('test sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/dev.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/nyt/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'dev.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('dev sentences', len(sentences))

    print('scierc')
    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/train.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    # print('all tags', valid_pattern['ner'] + valid_pattern['relation'])
    pot2file(save_path, 'train.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('train sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/test.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'test.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('test sentences', len(sentences))

    path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/dev.jsonl'
    save_path = '/data/liuweichang/Partially_Observed_TreeCRFs_ee/data/rel/scierc/pot'
    sentences, tags, valid_pattern  = coll2pot(path)
    pot2file(save_path, 'dev.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('dev sentences', len(sentences))