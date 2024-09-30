'''4.6将数据集写成一条一条的形式
ace 1,2 表示从1到2是一个span'''

import json

def get_data(file):
    json_data=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    return json_data

def dic_to_file(filename, dictionary):
	with open(filename, "w",encoding='utf-8') as outfile:
		outfile.write(json.dumps(dictionary, ensure_ascii=False, indent=4))

def get_ace(json_data):
    doc_key=[]
    sentences=[]
    ner=[]
    relations=[]
    events=[]
    sentence_start=[]
    for doc in json_data:
        doc_key.append(doc["doc_key"])
        sentences+=doc['sentences']
        ner+=doc['ner']
        relations+=doc['relations']
        events+=doc['events']
        sentence_start+=doc['sentence_start']
    return doc_key,sentences,ner,relations,events,sentence_start


def get_split(file):
    doc=[]
    with open(file,'r',encoding='utf-8') as f:
        dic=json.load(f)
    return dic

def get_ace_add(file,train_doc,test_doc,dev_doc):
    json_data=[]
    train_data=[]
    test_data=[]
    dev_data=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    for data in json_data:
        if data['doc_id'] in train_doc:
            train_data.append(data)
        elif data['doc_id'] in test_doc:
            test_data.append(data)
        else:
            dev_data.append(data)
    return train_data,test_data,dev_data

def print_res(sentences,ners,relations,events,sentence_starts):
    for sentence,ner,relation,event,sentence_start in zip(sentences,ners,relations,events,sentence_starts):
        if len(relation)==0:
            continue
        print(sentence)
        # print(ner)
        entity=[]

        for ner_ in ner:
            start,end,tag=ner_
            entity+=([','.join(sentence[start-sentence_start:end-sentence_start+1])+' '+tag])
        # print((entity))
        print(relation)

        rela=[]
        for relation_ in relation:
            start_1,end_1,start_2,end_2,tag=relation_
            rela+=([' '.join(sentence[min(start_1,start_2)-sentence_start:max(end_1,end_2)-sentence_start+1])+' '+tag])
        print('|'.join(rela))
        # print(event)
        eves=[]
        roles=[]
        for evens in event:
            for event_ in evens:
                if len(event_)==2:
                    pos,tag=event_
                    eves+=([''.join(sentence[pos-sentence_start])+' '+tag])
                elif len(event_)==3:
                    start,end,tag=event_
                    roles+=([''.join(sentence[start-sentence_start:end-sentence_start+1])+' '+tag])
        # print((eves))
        # print((roles))
        print(sentence_start)
        print()

if __name__=='__main__':
    train_path='ACE/train.json'
    test_path='ACE/test.json'
    dev_path='ACE/dev.json'
    train_data=get_data(train_path)
    doc_key,sentences,ners,relations,events,sentence_starts=get_ace(train_data)
    print_res(sentences,ners,relations,events,sentence_starts)

    print('test')
    test_data=get_data(test_path)
    doc_key,sentences,ners,relations,events,sentence_starts=get_ace(test_data)
    print_res(sentences,ners,relations,events,sentence_starts)
