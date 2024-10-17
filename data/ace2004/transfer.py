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
    sentences = dict()
    tags = []
    valid_pattern = dict()
    valid_pattern['ner'] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for line in lines:
            ners = []
            relations = []
            sentence = line['context']
            if sentence not in sentences:
                sentences[sentence] = []
            # print(line.keys())
            if line['entity_label'] not in valid_pattern['ner']:
                valid_pattern['ner'].append(line['entity_label'])

            for ner in line['span_position']:
                sentences[sentence].append(ner.split(';') + [line['entity_label']])

    return sentences.keys(), sentences.values(), valid_pattern

def pot2file(path, file, sentences, tags, valid_pattern, valid_pattern_path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + file, 'w', encoding='utf-8') as f:
        for sentence, tag in zip(sentences, tags):
            f.write(sentence + '\n')
            tmp = []
            for span in tag:
                tmp.append(str(span[0]) + ',' + str(int(span[1]) + 1) + ' ' + span[2])
            f.write('|'.join(tmp))
            f.write('\n')
            tmp = []
            for span in tag:
                tmp.append(str(span[0]) + ',' + str(int(span[1]) + 1) + ' ' + span[2])
            f.write('|'.join(tmp))
            f.write('\n\n')
    if 'train' in file:
        with open(valid_pattern_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(valid_pattern, ensure_ascii=False, indent=4))
        print('all tags', valid_pattern['ner'])
        print(len(valid_pattern['ner']))

if __name__ == '__main__':
    print('ace2004')
    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/mrc-ner.train'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    
    pot2file(save_path, 'train.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('train sentences', len(sentences))

    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/mrc-ner.test'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    
    pot2file(save_path, 'test.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('test sentences', len(sentences))

    path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/mrc-ner.dev'
    save_path = '/data/liuweichang/workspace/Partially_Observed_TreeCRFs_ee/data/ace2004/pot'
    valid_path = 'valid_pattern.json'
    sentences, tags, valid_pattern  = coll2pot(path)
    # print(len(valid_pattern['ner']), len(valid_pattern['relation']))
    
    pot2file(save_path, 'dev.txt', sentences, tags, valid_pattern, save_path + '/' + valid_path)
    print('dev sentences', len(sentences))