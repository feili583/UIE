'''4.4只提取关系和角色'''


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

def ace2pot(sentences,ner,relations,events,sentence_start):
    sen=[]
    tag=[]
    index=[]
    ner_tags=[]
    relation_tags=[]
    event_tags=[]
    role_tags=[]

    for id in range(len(sentences)):
        sen.append(' '.join(sentences[id]))
        ner_tag=[]
        ner_index=[]
        relation_tag=[]
        relation_index=[]
        event_tag=[]
        event_index=[]
        # for ner_ in ner[id]:
        #     # print(ner_)
        #     ner_tag.append(ner_[-1])
        #     ner_index.append(ner_[0:-1])
        #     ner_tags.append(ner_[-1].split('.')[-1])

        for relation in relations[id]:
            # print(relation)
            
            relation_index.append(relation[0:-1])
            print('Artifact'==relation[-1].split('.')[-1])
            if 'Artifact'==relation[-1].split('.')[-1]:
                relation_tags.append('re_Artifact')
                relation_tag.append('re_Artifact')
            else:
                relation_tags.append(relation[-1].split('.')[-1])
                relation_tag.append(relation[-1])
        # for event in events[id]:
        #     # print(event)
        #     event_tag.append(event[0][-1])
        #     event_index.append([event[0][0],event[0][0]])
        #     event_tags.append(event[0][-1].split('.')[-1])
        #     for id in range(1,len(event)):
                
        #         event_index.append([event[0][0],event[0][0]]+event[id][0:-1])
        #         if 'Artifact'==event[id][-1].split('.')[-1]:
        #             role_tags.append('eve_Artifact')
        #             event_tag.append('eve_Artifact')
        #         else:
        #             event_tag.append(event[id][-1])
        #             role_tags.append(event[id][-1].split('.')[-1])

        tag.append(ner_tag+relation_tag+event_tag)
        
        index.append(ner_index+relation_index+event_index)
    print('ner_tag:',set(ner_tags),'\n',len(ner_tags))
    print('relation_tag:',set(relation_tags),'\n',len(relation_tags))
    print('event_tags:',set(event_tags),'\n',len(event_tags))
    print('role_tags:',set(role_tags),'\n',len(role_tags))
    print('all_tags:',set(ner_tags+relation_tags+event_tags+role_tags),'\n',len(ner_tags+relation_tags+event_tags+role_tags))
    print(len(set(ner_tags+relation_tags+event_tags+role_tags)))
    print(len(set(role_tags).intersection(set(relation_tags))),set(role_tags).intersection(set(relation_tags)))
    return sen,tag,index,sentence_start

def pot2file(file,sen,tag,index,sentence_start):
    with open(file,'w',encoding='utf-8') as f:
        for id in range(len(sen)):
            # print(sen[id])
            # if len(tag[id])==0:
            #     # f.write('\n\n')
            #     continue

            f.write(sen[id]+'\n')
            f.write(sen[id]+'\n')
            if len(tag[id])==0:
                f.write('\n\n')
                continue
            tmp=[]
            # print(tag[id])
            # print(index[id])
            for idx in range(len(tag[id])):
                # print(index[id][idx])
                if len(index[id][idx])>2:
                    # enti1=min(index[id][idx][0],index[id][idx][1])
                    # enti2=min(index[id][idx][-1],index[id][idx][-2])
                    # rela_start=min(enti1,enti2)
                    # rela_end=max(enti1,enti2)
                
                    # tmp.append(str(rela_start-sentence_start[id])+','+str(rela_end-sentence_start[id]+1)+' '+tag[id][idx].split('.')[-1])
                    tmp.append(str(min(index[id][idx][0]-sentence_start[id],index[id][idx][-2]-sentence_start[id]))+','+str(max(index[id][idx][1]-sentence_start[id],index[id][idx][-1]-sentence_start[id])+1)+' '+tag[id][idx].split('.')[-1])
                else:
                    tmp.append(str(index[id][idx][0]-sentence_start[id])+','+str(index[id][idx][1]-sentence_start[id]+1)+' '+tag[id][idx].split('.')[-1])
            f.write('|'.join(tmp)+'\n\n')

def ace():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    pot2file('./ACE/pot_train_new.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/dev.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_dev.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    pot2file('./ACE/pot_dev_new.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/test.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_test.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    pot2file('./ACE/pot_test_new.txt',sen,tag,index,sentence_start)

def ace_relation():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,relations,temp_list,sentence_start)
    pot2file('./ACE_relation/pot_train.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/dev.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_dev.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,relations,temp_list,sentence_start)
    pot2file('./ACE_relation/pot_dev.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/test.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_test.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,relations,temp_list,sentence_start)
    pot2file('./ACE_relation/pot_test.txt',sen,tag,index,sentence_start)

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

def ace_add_pot(data):
    sentences=[]
    tags=[]
    ner_tags=[]
    relation_tags=[]
    event_tags=[]
    role_tags=[]

    for data_ in data:
        ners=[]
        relations=[]
        events=[]
        ner_ids={}
        ner_end_ids={}
        sentences.append(' '.join(data_['tokens']))
        for ner in data_['entity_mentions']:
            ner_ids[ner['id']]=ner['start']
            ner_end_ids[ner['id']]=ner['end']-1
            ners.append([ner['start'],ner['end']-1,ner['entity_subtype']])
            ner_tags.append(ner['entity_subtype'].split(':')[-1])
        for relation in data_['relation_mentions']:
            argument1_start=ner_ids[relation['arguments'][0]['entity_id']]
            argument2_start=ner_ids[relation['arguments'][1]['entity_id']]
            argument1_end=ner_end_ids[relation['arguments'][0]['entity_id']]
            argument2_end=ner_end_ids[relation['arguments'][1]['entity_id']]
            if 'Artifact'==relation['relation_subtype'].split(':')[-1]:
                relations.append([min(argument1_start,argument2_start),max(argument1_end,argument2_end),'re_Artifact'])
                relation_tags.append('re_Artifact')
            else:
                relations.append([min(argument1_start,argument2_start),max(argument1_end,argument2_end),relation['relation_subtype']])
                relation_tags.append(relation['relation_subtype'].split(':')[-1])
        for event in data_['event_mentions']:
            events.append([event['trigger']['start'],event['trigger']['end']-1,event['event_type']])
            event_tags.append(event['event_type'].split(':')[-1])
            for argu in event['arguments']:
                argu1_start=ner_ids[argu['entity_id']]
                argu1_end=ner_ids[argu['entity_id']]
                if 'Artifact'==argu['role'].split(':')[-1]:
                    events.append([min(argu1_start,event['trigger']['start']),max(argu1_end,event['trigger']['end']-1),'eve_Artifact'])
                    role_tags.append('eve_Artifact')
                else:
                    events.append([min(argu1_start,event['trigger']['start']),max(argu1_end,event['trigger']['end']-1),argu['role']])
                    role_tags.append(argu['role'].split(':')[-1])
        tags.append(ners+relations+events)

    print('ner_tag:',set(ner_tags),len(ner_tags))
    print('relation_tag:',set(relation_tags),len(relation_tags))
    print('event_tags:',set(event_tags),len(event_tags))
    print('role_tags:',set(role_tags),len(role_tags))
    print('all_tags:',set(ner_tags+relation_tags+event_tags+role_tags),len(ner_tags+relation_tags+event_tags+role_tags))
    print(len(set(ner_tags+relation_tags+event_tags+role_tags)))
    print(len(set(role_tags).intersection(set(relation_tags))),set(role_tags).intersection(set(relation_tags)))
    return sentences,tags

def ace_add_file(file,sentences,tags):
    with open(file,'w',encoding='utf-8') as f:
        for id in range(len(sentences)):
            if len(tags[id])==0:
                continue
            f.write(sentences[id]+'\n')
            f.write(sentences[id]+'\n')
            str_tags=[]
            for tag in tags[id]:
                string=str(tag[0])+','+str(tag[1]+1)+' '+tag[2].split(":")[-1]
                str_tags.append(string)
            f.write('|'.join(str_tags)+'\n\n')

def ace_add():
    train_doc=get_split('./ACE/doc_train.json')
    test_doc=get_split('./ACE/doc_test.json')
    dev_doc=get_split('./ACE/doc_dev.json')
    train_data,test_data,dev_data=get_ace_add('./ACE_add/english.oneie.json',train_doc,test_doc,dev_doc)
    train_sentences,train_tags=ace_add_pot(train_data)
    ace_add_file('./ACE_add/pot_train_new.txt',train_sentences,train_tags)

    test_sentences,test_tags=ace_add_pot(test_data)
    ace_add_file('./ACE_add/pot_test_new.txt',test_sentences,test_tags)

    dev_sentences,dev_tags=ace_add_pot(dev_data)
    ace_add_file('./ACE_add/pot_dev_new.txt',dev_sentences,dev_tags)

if __name__=='__main__':
    # print('ace')
    # ace()
    # print()
    # print('ace+')
    # ace_add()
    print('ace_relation')
    ace_relation()