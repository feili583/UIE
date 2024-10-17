'''2.25将ee数据处理成pot需要的格式
4,7 HCCX|4,7 HCCX
23.10.21加入数据集统计,允许加入空行,对于两个实体,选择min(实体头)+max(实体尾),使用二级标签
23.10.25将第二行修改为关系和论元所有可能的span
23.10.26将关系和事件分开
23.10.31只留下关系,只留下事件
23.12.24加入反向关系
24.10.13将实体从子类型改成类型'''

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
    tags_dict={}
    tags_dict['relation_entity']={}
    tags_dict['event_role']={}
    tags_dict['role_entity']={}

    for id in range(len(sentences)):
        sen.append(' '.join(sentences[id]))
        ner_tag=[]
        ner_index=[]
        relation_tag=[]
        relation_index=[]
        event_tag=[]
        event_index=[]
        ner_dict={}
        for ner_ in ner[id]:
            # print(ner_)
            ner_tag.append(ner_[-1])
            ner_index.append(ner_[0:-1])
            ner_tags.append(ner_[-1])
            ner_dict[str(ner_[0])+','+str(ner_[-2])]=ner_[-1]
        for relation in relations[id]:
            # print(relation)
            relation_tag.append(relation[-1].split('.')[0])
            relation_index.append(relation[0:-1])
            
            if relation[0] > relation[2]:
                relation_tags.append('r_'+relation[-1].split('.')[0])
                if 'r_'+relation[-1].split('.')[0] not in tags_dict['relation_entity'].keys():
                    tags_dict['relation_entity']['r_'+relation[-1].split('.')[0]]=[]
                tags_dict['relation_entity']['r_'+relation[-1].split('.')[0]].append(ner_dict[str(relation[0])+','+str(relation[1])])
                tags_dict['relation_entity']['r_'+relation[-1].split('.')[0]].append(ner_dict[str(relation[2])+','+str(relation[3])])
                tags_dict['relation_entity']['r_'+relation[-1].split('.')[0]]=list(set(tags_dict['relation_entity']['r_'+relation[-1].split('.')[0]]))
            else:
                relation_tags.append(relation[-1].split('.')[0])
                if relation[-1].split('.')[0] not in tags_dict['relation_entity'].keys():
                    tags_dict['relation_entity'][relation[-1].split('.')[0]]=[]
                tags_dict['relation_entity'][relation[-1].split('.')[0]].append(ner_dict[str(relation[0])+','+str(relation[1])])
                tags_dict['relation_entity'][relation[-1].split('.')[0]].append(ner_dict[str(relation[2])+','+str(relation[3])])
                tags_dict['relation_entity'][relation[-1].split('.')[0]]=list(set(tags_dict['relation_entity'][relation[-1].split('.')[0]]))
        for event in events[id]:
            event_tag.append(event[0][-1])
            event_index.append([event[0][0],event[0][0]])
            event_tags.append(event[0][-1])
            for id in range(1,len(event)):
                event_tag.append(event[id][-1])
                event_index.append([event[0][0],event[0][0]]+event[id][0:-1])
                if event[0][0] > event[id][0]:
                    role_tags.append('r_'+event[id][-1])
                    if event[0][-1] not in tags_dict['event_role'].keys():
                        tags_dict['event_role'][event[0][-1]]=[]
                    tags_dict['event_role'][event[0][-1]].append('r_'+event[id][-1])
                    tags_dict['event_role'][event[0][-1]]=list(set(tags_dict['event_role'][event[0][-1]]))
                    if 'r_'+event[id][-1] not in tags_dict['role_entity'].keys():
                        tags_dict['role_entity']['r_'+event[id][-1]]=[]
                    tags_dict['role_entity']['r_'+event[id][-1]].append(ner_dict[str(event[id][0])+','+str(event[id][1])])
                    tags_dict['role_entity']['r_'+event[id][-1]]=list(set(tags_dict['role_entity']['r_'+event[id][-1]]))
                else:
                    role_tags.append(event[id][-1])
                    if event[0][-1] not in tags_dict['event_role'].keys():
                        tags_dict['event_role'][event[0][-1]]=[]
                    tags_dict['event_role'][event[0][-1]].append(event[id][-1])
                    tags_dict['event_role'][event[0][-1]]=list(set(tags_dict['event_role'][event[0][-1]]))
                    if event[id][-1] not in tags_dict['role_entity'].keys():
                        tags_dict['role_entity'][event[id][-1]]=[]
                    tags_dict['role_entity'][event[id][-1]].append(ner_dict[str(event[id][0])+','+str(event[id][1])])
                    tags_dict['role_entity'][event[id][-1]]=list(set(tags_dict['role_entity'][event[id][-1]]))

        tag.append(ner_tag+relation_tag + event_tag)
        # tag.append(ner_tag)
        index.append(ner_index+relation_index + event_index)
        # index.append(ner_index)

    tags_dict['ner']=list(set(ner_tags))
    tags_dict['relation']=list(set(relation_tags))
    tags_dict['event']=list(set(event_tags))
    tags_dict['role']=list(set(role_tags))

    with open('./ACE/valid_pattern.json','w',encoding='utf-8') as f:
        f.write(json.dumps(tags_dict,ensure_ascii=False,indent=4))

    print('ner_tag:',len(ner_tags),len(set(ner_tags)), set(ner_tags))
    print('relation_tag:',len(relation_tags),len(set(relation_tags)), set(relation_tags))
    print('event_tags:',len(event_tags),len(set(event_tags)), set(event_tags))
    print('role_tags:',len(role_tags),len(set(role_tags)), set(role_tags))
    print('all_tags:',len(ner_tags+relation_tags+event_tags+role_tags),len(set(ner_tags+relation_tags+event_tags+role_tags)),set(ner_tags+relation_tags+event_tags+role_tags))
    print(len(set(ner_tags+relation_tags+event_tags+role_tags)))
    print(len(set(role_tags).intersection(set(relation_tags))),set(role_tags).intersection(set(relation_tags)))
    return sen,tag,index,sentence_start

def pot2file(file,sen,tag,index,sentence_start):
    with open(file,'w',encoding='utf-8') as f:
        for id in range(len(sen)):
            # print(sen[id])
            # if len(tag[id])==0:
            #     continue
            f.write(sen[id]+'\n')
            # f.write(sen[id]+'\n')
            tmp=[]
            all_tmp=[]
            # print(tag[id])
            # print(index[id])
            for idx in range(len(tag[id])):
                # print(index[id][idx])
                if len(index[id][idx])>2:
                    # enti1=min(index[id][idx][0],index[id][idx][1])
                    # enti2=min(index[id][idx][-1],index[id][idx][-2])
                    # rela_start=min(enti1,enti2)
                    # rela_end=max(enti1,enti2)
                
                    # tmp.append(str(rela_start-sentence_start[id])+','+str(rela_end-sentence_start[id]+1)+' '+tag[id][idx])
                    for id_start in range(index[id][idx][0]-sentence_start[id],index[id][idx][1]-sentence_start[id]+1):
                        for id_end in range(index[id][idx][-2]-sentence_start[id],index[id][idx][-1]-sentence_start[id]+1):
                            if id_start > id_end:
                                all_tmp.append(str(min(id_start,id_end))+','+str(max(id_start,id_end)+1)+' r_'+tag[id][idx])
                            else:
                                all_tmp.append(str(min(id_start,id_end))+','+str(max(id_start,id_end)+1)+' '+tag[id][idx])
                    if index[id][idx][0] > index[id][idx][-2]:
                        tmp.append(str(min(index[id][idx][0]-sentence_start[id],index[id][idx][-2]-sentence_start[id]))+','+str(max(index[id][idx][1]-sentence_start[id],index[id][idx][-1]-sentence_start[id])+1)+' r_'+tag[id][idx])
                    else:
                        tmp.append(str(min(index[id][idx][0]-sentence_start[id],index[id][idx][-2]-sentence_start[id]))+','+str(max(index[id][idx][1]-sentence_start[id],index[id][idx][-1]-sentence_start[id])+1)+' '+tag[id][idx])
                else:
                    tmp.append(str(index[id][idx][0]-sentence_start[id])+','+str(index[id][idx][1]-sentence_start[id]+1)+' '+tag[id][idx])
                    all_tmp.append(str(index[id][idx][0]-sentence_start[id])+','+str(index[id][idx][1]-sentence_start[id]+1)+' '+tag[id][idx])
            f.write('|'.join(all_tmp)+'\n')
            f.write('|'.join(tmp)+'\n\n')

def ace():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    pot2file('./ACE/pot_train_1224.txt',sen,tag,index,sentence_start)
    print(len(sentences)) 

    # ace_data=get_data('./ACE/dev.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_dev.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    # pot2file('./ACE/pot_dev_1224.txt',sen,tag,index,sentence_start)
    # print(len(sentences))

    # ace_data=get_data('./ACE/test.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_test.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,events,sentence_start)
    # pot2file('./ACE/pot_test_1224.txt',sen,tag,index,sentence_start)

    # print(len(sentences))

def ace_event():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,ner,temp_list,events,sentence_start)
    pot2file('./ACE_event/pot_train_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/dev.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_dev.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,temp_list,events,sentence_start)
    # pot2file('./ACE_event/pot_dev_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/test.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_test.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,temp_list,events,sentence_start)
    # pot2file('./ACE_event/pot_test_1224.txt',sen,tag,index,sentence_start)

def ace_relation():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    pot2file('./ACE_relation/pot_train_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/dev.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_dev.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_relation/pot_dev_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/test.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_test.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_relation/pot_test_1224.txt',sen,tag,index,sentence_start)

def ace_entity():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    pot2file('./ACE_entity/pot_train_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/dev.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_dev.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_entity/pot_dev_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/test.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_test.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_entity/pot_test_1224.txt',sen,tag,index,sentence_start)

def ace_only_relation():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    pot2file('./ACE_only_relation/pot_train_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/dev.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_dev.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_only_relation/pot_dev_1224.txt',sen,tag,index,sentence_start)

    # ace_data=get_data('./ACE/test.json')
    # doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    # dic_to_file('./ACE/doc_test.json',doc_key)
    # sen,tag,index,sentence_start=ace2pot(sentences,ner,relations,temp_list,sentence_start)
    # pot2file('./ACE_only_relation/pot_test_1224.txt',sen,tag,index,sentence_start)

def ace_only_event():
    ace_data=get_data('./ACE/train.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_train.json',doc_key)
    temp_list=[]
    for i in range(len(sentences)):
        temp_list.append([])
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,temp_list,events,sentence_start)
    pot2file('./ACE_only_event/pot_train_1031.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/dev.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_dev.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,temp_list,events,sentence_start)
    pot2file('./ACE_only_event/pot_dev_1031.txt',sen,tag,index,sentence_start)

    ace_data=get_data('./ACE/test.json')
    doc_key,sentences,ner,relations,events,sentence_start=get_ace(ace_data)
    dic_to_file('./ACE/doc_test.json',doc_key)
    sen,tag,index,sentence_start=ace2pot(sentences,temp_list,temp_list,events,sentence_start)
    pot2file('./ACE_only_event/pot_test_1031.txt',sen,tag,index,sentence_start)

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
    a=0
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    for data in json_data:
        if data['doc_id'] in train_doc:
            train_data.append(data)
        elif data['doc_id'] in test_doc:
            test_data.append(data)
        elif data['doc_id'] in dev_doc:
            dev_data.append(data)
        else:
            a+=1
            # print(data['doc_id'])
            # print(data)
    print(a)
    with open('ACE_add/train_english.oneie.json','w',encoding='utf-8') as f:
        for data in train_data:
            f.write(json.dumps(data)+'\n')

    with open('ACE_add/test_english.oneie.json','w',encoding='utf-8') as f:
        for data in test_data:
            f.write(json.dumps(data))
    with open('ACE_add/dev_english.oneie.json','w',encoding='utf-8') as f:
        for data in dev_data:
            f.write(json.dumps(data))
    print(len(train_data),len(test_data),len(dev_data))
    return train_data,test_data,dev_data

def ace_add_pot(data):
    sentences=[]
    tags=[]
    all_tags=[]
    tags_dict={}
    ner_tags=[]
    relation_tags=[]
    event_tags=[]
    role_tags=[]

    tags_dict['ner']=[]
    tags_dict['event']=[]
    tags_dict['relation']=[]
    tags_dict['role']=[]
    tags_dict['role_entity']={}
    tags_dict['relation_entity']={}
    tags_dict['event_role']={}

    for data_ in data:
        ners=[]
        relations=[]
        all_relaitons=[]
        events=[]
        all_events=[]
        ner_ids={}
        ner_types={}
        ner_end_ids={}
        sentences.append(' '.join(data_['tokens']))
        for ner in data_['entity_mentions']:
            ner_ids[ner['id']]=ner['start']
            ner_end_ids[ner['id']]=ner['end']-1
            ner_types[ner['id']]=ner['entity_type']
            ners.append([ner['start'],ner['end']-1,ner['entity_type']])
            ner_tags.append(ner['entity_type'])
        for relation in data_['relation_mentions']:
            argument1=ner_ids[relation['arguments'][0]['entity_id']]
            argument1_end=ner_end_ids[relation['arguments'][0]['entity_id']]
            argument2=ner_ids[relation['arguments'][1]['entity_id']]
            argument2_end=ner_end_ids[relation['arguments'][1]['entity_id']]
            if argument1 > argument2:
                relations.append([min(argument1,argument2),max(argument1_end,argument2_end),'r_'+relation['relation_type']])
            else:
                relations.append([min(argument1,argument2),max(argument1_end,argument2_end),relation['relation_type']])
            for id_start in range(argument1,argument1_end+1):
                for id_end in range(argument2,argument2_end+1):
                    if id_start > id_end:
                        all_relaitons.append([min(id_start,id_end),max(id_start,id_end),'r_'+relation['relation_type']])
                        relation_tags.append('r_'+relation['relation_type'])
                    else:
                        all_relaitons.append([min(id_start,id_end),max(id_start,id_end),relation['relation_type']])
                        relation_tags.append(relation['relation_type'])
            
            if relation['relation_type'] not in tags_dict['relation_entity'].keys():
                tags_dict['relation_entity'][relation['relation_type']]=[]
            tags_dict['relation_entity'][relation['relation_type']].append(ner_types[relation['arguments'][0]['entity_id']])
            tags_dict['relation_entity'][relation['relation_type']]=list(set(tags_dict['relation_entity'][relation['relation_type']]))
        for event in data_['event_mentions']:
            events.append([event['trigger']['start'],event['trigger']['end']-1,event['event_type']])
            all_events.append([event['trigger']['start'],event['trigger']['end']-1,event['event_type']])
            event_tags.append(event['event_type'])
            for argu in event['arguments']:
                argu1=ner_ids[argu['entity_id']]
                argu1_end=ner_end_ids[argu['entity_id']]
                if argu1 < event['trigger']['start']:
                    events.append([min(argu1,event['trigger']['start']),max(argu1_end,event['trigger']['end']-1),'r_'+argu['role']])
                    role_tags.append('r_'+argu['role'])
                else:
                    events.append([min(argu1,event['trigger']['start']),max(argu1_end,event['trigger']['end']-1),argu['role']])
                    role_tags.append(argu['role'])
                for id_start in range(argu1, argu1_end+1):
                    for id_end in range(event['trigger']['start'],event['trigger']['end']):
                        if id_start < id_end:
                            all_events.append([min(id_start,id_end),max(id_start,id_end),'r_'+argu['role']])
                        else:
                            all_events.append([min(id_start,id_end),max(id_start,id_end),argu['role']])
                
                if event['event_type'] not in tags_dict['event_role'].keys():
                    tags_dict['event_role'][event['event_type']]=[]
                tags_dict['event_role'][event['event_type']].append(argu['role'])
                tags_dict['event_role'][event['event_type']]=list(set(tags_dict['event_role'][event['event_type']]))
                if argu['role'] not in tags_dict['role_entity'].keys():
                    tags_dict['role_entity'][argu['role']]=[]
                tags_dict['role_entity'][argu['role']].append(ner_types[argu['entity_id']])
                tags_dict['role_entity'][argu['role']]=list(set(tags_dict['role_entity'][argu['role']]))
        # tags.append(ners+relations+events)
        # all_tags.append(ners+all_relaitons+all_events)
        tags.append(ners)
        all_tags.append(ners)

    tags_dict['ner']=list(set((ner_tags)))
    tags_dict['event']=list(set(event_tags))
    tags_dict['relation']=list(set(relation_tags))
    tags_dict['role']=list(set(role_tags))

    with open('./ACE_add_entity/valid_pattern.json','w',encoding='utf-8') as f:
        f.write(json.dumps(tags_dict,ensure_ascii=False,indent=4))

    print('ner_tag:',len(ner_tags),len(set(ner_tags)))
    print('relation_tag:',len(relation_tags),len(set(relation_tags)))
    print('event_tags:',len(event_tags),len(set(event_tags)))
    print('role_tags:',len(role_tags),len(set(role_tags)))
    print('all_tags:',len(ner_tags+relation_tags+event_tags+role_tags),set(ner_tags+relation_tags+event_tags+role_tags))
    print(len(set(ner_tags+relation_tags+event_tags+role_tags)))
    print(len(set(role_tags).intersection(set(relation_tags))),set(role_tags).intersection(set(relation_tags)))
    return sentences,tags, all_tags

def ace_add_file(file,sentences,tags, all_tags):
    with open(file,'w',encoding='utf-8') as f:
        for id in range(len(sentences)):
            # if len(tags[id])==0:
            #     continue
            f.write(sentences[id]+'\n')

            str_tags=[]
            for tag in all_tags[id]:
                string=str(tag[0])+','+str(tag[1]+1)+' '+tag[2]
                str_tags.append(string)
            f.write('|'.join(str_tags)+'\n')

            str_tags=[]
            for tag in tags[id]:
                string=str(tag[0])+','+str(tag[1]+1)+' '+tag[2]
                str_tags.append(string)
            f.write('|'.join(str_tags)+'\n\n')

def ace_add():
    train_doc=get_split('./ACE/doc_train.json')
    test_doc=get_split('./ACE/doc_test.json')
    dev_doc=get_split('./ACE/doc_dev.json')
    train_data,test_data,dev_data=get_ace_add('./ACE_add/english.oneie.json',train_doc,test_doc,dev_doc)
    train_sentences,train_tags, train_all_tags=ace_add_pot(train_data)
    ace_add_file('./ACE_add/pot_train_1224.txt',train_sentences,train_tags, train_all_tags)

    # test_sentences,test_tags, test_all_tags=ace_add_pot(test_data)
    # ace_add_file('./ACE_add/pot_test_1224.txt',test_sentences,test_tags, test_all_tags)

    # dev_sentences,dev_tags, dev_all_tags=ace_add_pot(dev_data)
    # ace_add_file('./ACE_add/pot_dev_1224.txt',dev_sentences,dev_tags, dev_all_tags)

def ace_add_relation():
    train_doc=get_split('./ACE/doc_train.json')
    test_doc=get_split('./ACE/doc_test.json')
    dev_doc=get_split('./ACE/doc_dev.json')
    train_data,test_data,dev_data=get_ace_add('./ACE_add/english.oneie.json',train_doc,test_doc,dev_doc)
    train_sentences,train_tags, train_all_tags=ace_add_pot(train_data)
    ace_add_file('./ACE_add_relation/pot_train_1224.txt',train_sentences,train_tags, train_all_tags)

    # test_sentences,test_tags, test_all_tags=ace_add_pot(test_data)
    # ace_add_file('./ACE_add_relation/pot_test_1224.txt',test_sentences,test_tags, test_all_tags)

    # dev_sentences,dev_tags, dev_all_tags=ace_add_pot(dev_data)
    # ace_add_file('./ACE_add_relation/pot_dev_1224.txt',dev_sentences,dev_tags, dev_all_tags)

def ace_add_event():
    train_doc=get_split('./ACE/doc_train.json')
    test_doc=get_split('./ACE/doc_test.json')
    dev_doc=get_split('./ACE/doc_dev.json')
    train_data,test_data,dev_data=get_ace_add('./ACE_add/english.oneie.json',train_doc,test_doc,dev_doc)
    train_sentences,train_tags, train_all_tags=ace_add_pot(train_data)
    ace_add_file('./ACE_add_event/pot_train_1224.txt',train_sentences,train_tags, train_all_tags)

    # test_sentences,test_tags, test_all_tags=ace_add_pot(test_data)
    # ace_add_file('./ACE_add_event/pot_test_1224.txt',test_sentences,test_tags, test_all_tags)

    # dev_sentences,dev_tags, dev_all_tags=ace_add_pot(dev_data)
    # ace_add_file('./ACE_add_event/pot_dev_1224.txt',dev_sentences,dev_tags, dev_all_tags)

def ace_add_entity():
    train_doc=get_split('./ACE/doc_train.json')
    test_doc=get_split('./ACE/doc_test.json')
    dev_doc=get_split('./ACE/doc_dev.json')
    train_data,test_data,dev_data=get_ace_add('./ACE_add/english.oneie.json',train_doc,test_doc,dev_doc)
    train_sentences,train_tags, train_all_tags=ace_add_pot(train_data)
    ace_add_file('./ACE_add_entity/pot_train_1224.txt',train_sentences,train_tags, train_all_tags)

    test_sentences,test_tags, test_all_tags=ace_add_pot(test_data)
    ace_add_file('./ACE_add_entity/pot_test_1224.txt',test_sentences,test_tags, test_all_tags)

    dev_sentences,dev_tags, dev_all_tags=ace_add_pot(dev_data)
    ace_add_file('./ACE_add_entity/pot_dev_1224.txt',dev_sentences,dev_tags, dev_all_tags)

if __name__=='__main__':
    # print('ace')
    # ace()
    # print()
    # print('ace+')
    # ace_add()
    # print('ace_relation')
    # ace_relation()
    # print('ace_add_relation')
    # ace_add_relation()
    print('ace_event')
    ace_event()
    # print('ace_add_event')
    # ace_add_event()
    # print('ace_only_event')
    # ace_only_event()
    # print('ace_only_relation')
    # ace_only_relation()
    # print('ace_add_entity')
    # ace_add_entity()
    # print('ace_entity')
    # ace_entity()