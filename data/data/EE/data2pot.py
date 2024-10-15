import json
#实体、关系、事件、触发词、事件元素
'''所有的index，pot需要结束位置均+1,但是原数据集ACE已经+1，因此只需要关系和event结束位置+1
12.4加入不含有关系的二级标签，修改事件的标注，trigger的标签是event_subtype，trigger到两个论元(实体)的开始位置的标签是role标签'''
def dic_to_file(filename, dictionary):
	with open(filename, "w",encoding='utf-8') as outfile:
		outfile.write(json.dumps(dictionary, ensure_ascii=False, indent=4))

def test():
    with open('./ACE+/english.oneie.json','r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
        dic_to_file('./ACE+/copy.oneie.json',json_data)

def get_index_first(file):
    '''关系使用一级标签，并关系标签翻转'''
    with open(file,'r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    sentences=[]
    doc_id=[]
    sent_id=[]
    entities=[]
    triggers=[]
    relations=[]
    events=[]
    types=[]
    trigger_types=['trigger']
    entity_types=[]
    relation_types=[]#两个实体
    event_types=[]#trigger和两个argument
    '''实体1个F1，关系1个F1，事件的trigger+argument(2个)1个F1，事件的argument1个F1'''
    types.append('trigger')
    for sen in json_data:
        sent_id.append(sen['sent_id'])
        doc_id.append(sen['doc_id'])
        sentences.append(' '.join(sen['tokens']))
        enti=[]
        for entity in sen['entity_mentions']:
            enti.append({entity['id']:[entity['start'],entity['end'],entity['entity_type']]})
            types.append(entity['entity_type'])
            entity_types.append(entity['entity_type'])
        entities.append(enti)
        rela=[]
        for relation in sen['relation_mentions']:
            index=[]
            relation_types_single=relation['relation_type']
            for argu in relation['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append(item[argu['entity_id']][0])
            if len(index)==2:
                if index[0]<index[1]:
                    rela.append([min(index[0],index[1]),max(index[0],index[1])+1,relation['relation_type']])
                else:
                    rela.append([min(index[0],index[1]),max(index[0],index[1])+1,'r'+relation['relation_type']])
                    relation_types_single='r'+relation_types_single
            types.append(relation_types_single)
            relation_types.append(relation_types_single)
        relations.append(rela)
        eve=[]
        trigger=[]
        for event in sen['event_mentions']:
            trigger.append([event['trigger']['start'],event['trigger']['end'],'trigger'])
            index=[]
            for argu in event['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append(item[argu['entity_id']][0])
            if len(index)>0:
                for id in range(len(index)):
                    types_trigger_tmp=event['event_type']
                    if event['trigger']['start']<index[id]:
                        eve.append([min(event['trigger']['start'],index[id]),max(event['trigger']['start'],index[id])+1,event['event_type']])
                    else:
                        eve.append([min(event['trigger']['start'],index[id]),max(event['trigger']['start'],index[id])+1,'r'+event['event_type']])
                        types_trigger_tmp='r'+event['event_type']
                    types.append(types_trigger_tmp)
                    event_types.append(types_trigger_tmp)
        triggers.append(trigger)
        events.append(eve)
    print('trigger_types:',list(set(trigger_types)))
    print('entity_types:',list(set(entity_types)))
    print('relation_types:',list(set(relation_types)))
    print('event_types:',list(set(event_types)))

    return sentences,doc_id,sent_id,entities,triggers,relations,events,types

def check(sentences,doc_id,sent_id,entities,triggers,relations,events,types):
    for id in range(len(sentences)):
        for rela in relations[id]:
            entity_1=rela[0]
            entity_2=rela[1]-1
            flag=False
            for entity in entities[id]:
                if list(entity.values())[0][0]==entity_1:
                    flag=True
                    break
            if flag==False:
                print(sentences[id])
                print(rela)
                print(entities[id])
            flag=False
            for entity in entities[id]:
                if list(entity.values())[0][0]==entity_2:
                    flag=True
                    break
            if flag==False:
                print(sentences[id])
                print(rela)
                print(entities[id])
                    

def get_index_second(file):
    '''关系使用二级标签,并使用标签翻转'''
    with open(file,'r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    sentences=[]
    doc_id=[]
    sent_id=[]
    entities=[]
    triggers=[]
    relations=[]
    events=[]
    types=[]
    trigger_types=['trigger']
    entity_types=[]
    relation_types=[]#两个实体
    event_types=[]#trigger和两个argument
    '''实体1个F1，关系1个F1，事件的trigger+argument(2个)1个F1，事件的argument1个F1'''
    types.append('trigger')
    for sen in json_data:
        sent_id.append(sen['sent_id'])
        doc_id.append(sen['doc_id'])
        sentences.append(' '.join(sen['tokens']))
        enti=[]
        for entity in sen['entity_mentions']:
            enti.append({entity['id']:[entity['start'],entity['end'],entity['entity_subtype']]})
            types.append(entity['entity_subtype'])
            entity_types.append(entity['entity_subtype'])
        entities.append(enti)
        rela=[]
        for relation in sen['relation_mentions']:
            index=[]
            for argu in relation['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append(item[argu['entity_id']][0])
            if len(index)==2:
                relation_types_tmp=relation['relation_subtype']
                if index[0]<index[1]:
                    rela.append([min(index[0],index[1]),max(index[0],index[1])+1,relation['relation_subtype']])
                else:
                    rela.append([min(index[0],index[1]),max(index[0],index[1])+1,'r'+relation['relation_subtype']])
                    relation_types_tmp='r'+relation_types_tmp
            types.append(relation_types_tmp)
            relation_types.append(relation_types_tmp)
        relations.append(rela)
        eve=[]
        trigger=[]
        for event in sen['event_mentions']:
            trigger.append([event['trigger']['start'],event['trigger']['end'],'trigger'])
            index=[]
            for argu in event['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append(item[argu['entity_id']][0])
            if len(index)>0:
                for id in range(len(index)):
                    event_type_tmp=event['event_type']
                    if event['trigger']['start']<index[id]:
                        eve.append([min(event['trigger']['start'],index[id]),max(event['trigger']['start'],index[id])+1,event['event_type']])
                    else:
                        eve.append([min(event['trigger']['start'],index[id]),max(event['trigger']['start'],index[id])+1,'r'+event['event_type']])
                        event_type_tmp='r'+event_type_tmp
                    types.append(event_type_tmp)
                    event_types.append(event_type_tmp)
        triggers.append(trigger)
        events.append(eve)
    print('trigger_types:',list(set(trigger_types)))
    print('entity_types:',list(set(entity_types)))
    print('relation_types:',list(set(relation_types)))
    print('event_types:',list(set(event_types)))

    return sentences,doc_id,sent_id,entities,triggers,relations,events,types

def get_index_all_relations(file):
    '''标注所有可能的二级关系，不再只是取两个实体的开始位置作为两个实体的关系，把所有的位置都认为是有关系,并使用标签翻转'''
    with open(file,'r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    sentences=[]
    doc_id=[]
    sent_id=[]
    entities=[]
    triggers=[]
    relations=[]
    events=[]
    types=[]
    trigger_types=['trigger']
    entity_types=[]
    relation_types=[]#两个实体
    event_types=[]#trigger和两个argument
    '''实体1个F1，关系1个F1，事件的trigger+argument(2个)1个F1，事件的argument1个F1'''
    types.append('trigger')
    for sen in json_data:
        sent_id.append(sen['sent_id'])
        doc_id.append(sen['doc_id'])
        sentences.append(' '.join(sen['tokens']))
        enti=[]
        for entity in sen['entity_mentions']:
            enti.append({entity['id']:[entity['start'],entity['end'],entity['entity_subtype']]})
            types.append(entity['entity_subtype'])
            entity_types.append(entity['entity_subtype'])
        entities.append(enti)
        rela=[]
        for relation in sen['relation_mentions']:
            index=[]
            for argu in relation['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append([item[argu['entity_id']][0],item[argu['entity_id']][1]])
            if len(index)==2:
                for s_i in range(index[0][0],index[0][1]):
                    for e_i in range(index[1][0],index[1][1]):
                        relation_types_tmp=relation['relation_subtype']
                        if s_i<e_i:
                            rela.append([min(s_i,e_i),max(s_i,e_i)+1,relation['relation_subtype']])
                        else:
                            rela.append([min(s_i,e_i),max(s_i,e_i)+1,'r'+relation['relation_subtype']])
                            relation_types_tmp='r'+relation_types_tmp

                        types.append(relation_types_tmp)
                        relation_types.append(relation_types_tmp)
        relations.append(rela)
        eve=[]
        trigger=[]
        for event in sen['event_mentions']:
            trigger.append([event['trigger']['start'],event['trigger']['end'],'trigger'])
            index=[]
            for argu in event['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append([item[argu['entity_id']][0],item[argu['entity_id']][1]])
            if len(index)>0:
                for s_i in range(event['trigger']['start'],event['trigger']['end']):
                    for j in range(len(index)):
                        for e_i in range(index[j][0],index[j][1]):
                            types_tmp=event['event_type']
                            if s_i<e_i:
                                eve.append([min(s_i,e_i),max(s_i,e_i)+1,event['event_type']])
                            else:
                                eve.append([min(s_i,e_i),max(s_i,e_i)+1,'r'+event['event_type']])
                                types_tmp='r'+types_tmp

                            types.append(types_tmp)
                            event_types.append(types_tmp)
        triggers.append(trigger)
        events.append(eve)
    print('trigger_types:',list(set(trigger_types)))
    print('entity_types:',list(set(entity_types)))
    print('relation_types:',list(set(relation_types)))
    print('event_types:',list(set(event_types)))

    return sentences,doc_id,sent_id,entities,triggers,relations,events,types

def get2new_relation(file):
    '''12.5加入不含有关系的二级标签，修改事件的标注，trigger的标签是event_subtype，trigger到两个论元(实体)的开始位置的标签是role标签'''
    with open(file,'r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    sentences=[]
    doc_id=[]
    sent_id=[]
    entities=[]
    triggers=[]
    relations=[]
    events=[]
    types=[]
    trigger_types=[]#event
    entity_types=[]
    relation_types=[]#两个实体
    event_types=[]#role类型

    for sen in json_data:
        sent_id.append(sen['sent_id'])
        doc_id.append(sen['doc_id'])
        sentences.append(' '.join(sen['tokens']))
        enti=[]
        for entity in sen['entity_mentions']:
            enti.append({entity['id']:[entity['start'],entity['end'],entity['entity_subtype']]})
            types.append(entity['entity_subtype'])
            entity_types.append(entity['entity_subtype'])
        entities.append(enti)
        # rela=[]
        # for relation in sen['relation_mentions']:
        #     index=[]
        #     for argu in relation['arguments']:
        #         for item in enti:
        #             if argu['entity_id'] in item.keys():
        #                 index.append([item[argu['entity_id']][0],item[argu['entity_id']][1]])
        #     if len(index)==2:
        #         for s_i in range(index[0][0],index[0][1]):
        #             for e_i in range(index[1][0],index[1][1]):
        #                 relation_types_tmp=relation['relation_subtype']
        #                 if s_i<e_i:
        #                     rela.append([min(s_i,e_i),max(s_i,e_i)+1,relation['relation_subtype']])
        #                 else:
        #                     rela.append([min(s_i,e_i),max(s_i,e_i)+1,'r'+relation['relation_subtype']])
        #                     relation_types_tmp='r'+relation_types_tmp

        #                 types.append(relation_types_tmp)
        #                 relation_types.append(relation_types_tmp)
        # relations.append(rela)
        eve=[]
        trigger=[]
        for event in sen['event_mentions']:
            trigger_tmp_type=event['event_type'].split(':')[1]
            trigger.append([event['trigger']['start'],event['trigger']['end'],trigger_tmp_type])
            trigger_types.append(trigger_tmp_type)
            types.append(trigger_tmp_type)
            index=[]
            for argu in event['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append([item[argu['entity_id']][0],item[argu['entity_id']][1],argu['role']])
            if len(index)>0:
                '''事件所有位置的实体'''
                # for s_i in range(event['trigger']['start'],event['trigger']['end']):
                #     for j in range(len(index)):
                #         for e_i in range(index[j][0],index[j][1]):
                #             types_tmp=index[j][2]
                #             if s_i<e_i:
                #                 eve.append([min(s_i,e_i),max(s_i,e_i)+1,types_tmp])
                #             else:
                #                 eve.append([min(s_i,e_i),max(s_i,e_i)+1,'r'+types_tmp])
                #                 types_tmp='r'+types_tmp

                #             types.append(types_tmp)
                #             event_types.append(types_tmp)

                for j in range(len(index)):
                    s_i=event['trigger']['start']
                    e_i=index[j][0]
                    types_tmp=index[j][2]
                    if s_i<e_i:
                        eve.append([min(s_i,e_i),max(s_i,e_i)+1,types_tmp])
                    else:
                        eve.append([min(s_i,e_i),max(s_i,e_i)+1,'r'+types_tmp])
                        types_tmp='r'+types_tmp

                    types.append(types_tmp)
                    event_types.append(types_tmp)
        triggers.append(trigger)
        events.append(eve)
    print('trigger_types:',list(set(trigger_types)))
    print('entity_types:',list(set(entity_types)))
    # print('relation_types:',list(set(relation_types)))
    print('role_types:',list(set(event_types)))
    return sentences,doc_id,sent_id,entities,triggers,relations,events,types

def event2pot(file):
    '''12.5加入不含有关系的二级标签，修改事件的标注，trigger的标签是event_subtype，trigger到两个论元(实体)的开始位置的标签是role标签'''
    with open(file,'r',encoding='utf-8') as f:
        json_data=[]
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    sentences=[]
    doc_id=[]
    sent_id=[]
    entities=[]
    triggers=[]
    relations=[]
    events=[]
    types=[]
    trigger_types=[]#event
    entity_types=[]
    relation_types=[]#两个实体
    event_types=[]#role类型

    for sen in json_data:
        sent_id.append(sen['sent_id'])
        doc_id.append(sen['doc_id'])
        sentences.append(' '.join(sen['tokens']))
        enti=[]
        for entity in sen['entity_mentions']:
            enti.append({entity['id']:[entity['start'],entity['end'],entity['entity_subtype']]})
            types.append(entity['entity_subtype'])
            entity_types.append(entity['entity_subtype'])
        entities.append([])
        eve=[]
        trigger=[]
        for event in sen['event_mentions']:
            trigger_tmp_type=event['event_type'].split(':')[1]
            trigger.append([event['trigger']['start'],event['trigger']['end'],trigger_tmp_type])
            trigger_types.append(trigger_tmp_type)
            types.append(trigger_tmp_type)
            index=[]
            for argu in event['arguments']:
                for item in enti:
                    if argu['entity_id'] in item.keys():
                        index.append([item[argu['entity_id']][0],item[argu['entity_id']][1],argu['role']])
            if len(index)>0:
                for j in range(len(index)):
                    s_i=event['trigger']['start']
                    e_i=index[j][0]
                    types_tmp=index[j][2]
                    if s_i<e_i:
                        eve.append([min(s_i,e_i),max(s_i,e_i)+1,types_tmp])
                    else:
                        eve.append([min(s_i,e_i),max(s_i,e_i)+1,types_tmp])
                        types_tmp=types_tmp

                    types.append(types_tmp)
                    event_types.append(types_tmp)
        triggers.append(trigger)
        events.append(eve)
    print('trigger_types:',list(set(trigger_types)))
    print('entity_types:',list(set(entity_types)))
    # print('relation_types:',list(set(relation_types)))
    print('role_types:',list(set(event_types)))
    return sentences,doc_id,sent_id,entities,entities,entities,events,types

def divide(file):
    doc_ids=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            dic=json.loads(line)
            if dic['doc_key'] not in doc_ids:
                doc_ids.append(dic['doc_key'])
    return doc_ids
    

def index2pot(sentences,doc_id,sen_id,entities,triggers,relations,events,ids):
    sens=[]
    tags=[]

    for id in ids:
        for idx in range(len(doc_id)):
            if doc_id[idx]==id:
                # print(id)
                sens.append(sentences[idx])
                tmp=[]
                for entity in entities[idx]:
                    tmp.append(list(entity.values())[0])
                for trigger in triggers[idx]:
                    tmp.append(trigger)
                if len(relations)!=0:
                    for relation in relations[idx]:
                        tmp.append(relation)
                for event in events[idx]:
                    tmp.append(event)
                tags.append(tmp)
    return sens,tags

def write2file(sens,tags,file):
    sum=0
    with open(file,'w',encoding='utf-8') as f:
        for id in range(len(sens)):
            if len(tags[id])==0:
                continue
            f.write(sens[id]+'\n')
            f.write(sens[id]+'\n')
            sum+=1
            for idx in range(len(tags[id])):
                tag=tags[id][idx]
                if idx==len(tags[id])-1:
                    f.write(str(tag[0])+','+str(tag[1])+' '+tag[2]+'\n')
                else:
                    f.write(str(tag[0])+','+str(tag[1])+' '+tag[2]+'|')
            f.write('\n')
    print('数据集数量',sum)   

def get_from_ACE(file):
    json_data=[]
    all_sentences=[]
    all_ners=[]
    all_relations=[]
    all_events=[]
    all_sentence_start=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            dic=json.loads(line)
            json_data.append(dic)
    for item in json_data:
        for sentences in item['sentences']:
            all_sentences.append(sentences)

        for ners in item['ner']:
            ner_tmp=[]
            for ner in ners:
                ner_tmp.append(ner)
            all_ners.append(ner_tmp)
        for relations in item['relations']:
            relation_tmp=[]
            for relation in relations:
                relation_tmp.append(relation)
            all_relations.append(relation_tmp)
        for events in item['events']:
            event_tmps=[]
            for event in events:
                event_tmp=[]
                for event_ in event:
                    event_tmp.append(event_)
                event_tmps.append(event_tmp)
            all_events.append(event_tmps)
        all_sentence_start+=item['sentence_start']

    return all_sentences,all_ners,all_relations,all_events,all_sentence_start

def json2pot(all_sentences,all_ners,all_relations,all_events,all_sentence_start):
    absolute_ners=[]
    absolute_relations=[]
    absolute_events=[]
    absolute_triggers=[]
    absolute_sentences=[]
    all_types=[]
    ner_types=[]
    relation_types=[]
    event_types=[]
    trigger_types=[]

    for sentences,ners,relations,events,sentence_starts in zip(all_sentences,all_ners,all_relations,all_events,all_sentence_start):
        absolute_sentences.append(' '.join(sentences))

        tmp_absolute_ners=[]
        tmp_absolute_relations=[]
        tmp_absolute_events=[]
        tmp_absolute_triggers=[]
        for ner in ners:

            tmp_absolute_ners.append([ner[0]-sentence_starts,ner[1]-sentence_starts+1,ner[2]])
            ner_types.append(ner[2])
            all_types.append(ner[2])
        # for relation in relations:
        #     type_tmp=relation[4]
        #     if relation[0]<relation[2]:
        #         for start in range(relation[0],relation[1]+1):
        #             for end in range(relation[2],relation[3]+1):
        #                 tmp_absolute_relations.append([start-sentence_starts,end-sentence_starts+1,type_tmp])
        #     else:
        #         for start in range(relation[2],relation[3]+1):
        #             for end in range(relation[0],relation[1]+1):
        #                 tmp_absolute_relations.append([start-sentence_starts,end-sentence_starts+1,'r'+type_tmp])
        #         type_tmp='r'+type_tmp
        #     relation_types.append(type_tmp)
        #     all_types.append(type_tmp)
        for event_ in events:
            tmp_absolute_trigger=[]
            for event in event_:
                if len(event)==2:
                    tmp_absolute_trigger.append(event[0])
                    tmp_absolute_triggers.append([event[0]-sentence_starts,event[0]-sentence_starts+1,event[1]])
                    trigger_types.append(event[1])
                    all_types.append(event[1])
                else:
                    if tmp_absolute_trigger[0]<event[0]:
                        for start in range(tmp_absolute_trigger[0]-sentence_starts,tmp_absolute_trigger[0]-sentence_starts+1):
                            for end in range(event[0]-sentence_starts,event[1]-sentence_starts+1):
                                tmp_absolute_events.append([start,end+1,event[2]])
                                event_types.append(event[2])
                                all_types.append(event[2])
                    else:
                        for start in range(event[0]-sentence_starts,event[1]-sentence_starts+1):
                            for end in range(tmp_absolute_trigger[0]-sentence_starts,tmp_absolute_trigger[0]-sentence_starts+1):
                                tmp_absolute_events.append([start,end+1,'r'+event[2]])
                                event_types.append('r'+event[2])
                                all_types.append('r'+event[2])
        absolute_ners.append(tmp_absolute_ners)
        absolute_relations.append(tmp_absolute_relations)
        absolute_events.append(tmp_absolute_events)
        absolute_triggers.append(tmp_absolute_triggers)
    print('ner类型',list(set(ner_types)))
    print('trigger类型',list(set(trigger_types)))
    print('relation类型',list(set(relation_types)))
    print('角色类型',list(set(event_types)))
    print('全部类型',list(set(all_types)))

    return absolute_sentences,absolute_ners,absolute_triggers,absolute_relations,absolute_events

def sen2tag(sentences,absolute_ners,absolute_triggers,absolute_relations,absolute_events):
    all_tags=[]
    for ners,triggers,relations,events in zip(absolute_ners,absolute_triggers,absolute_relations,absolute_events):
        tags=[]
        for ner in ners:
            # print(ner)
            tags.append(ner)
        for trigger in triggers:
            # print(trigger)
            tags.append(trigger)
        for relation in relations:
            # print(relation)
            tags.append(relation)
        for event in events:
            # print(event)
            tags.append(event)
        all_tags.append(tags)
    
    return sentences,all_tags

def ACE2pot(file,save_file):
    print('读取文件')
    all_sentences,all_ners,all_relations,all_events,all_sentence_start=get_from_ACE(file)
    print('计算成pot形式')
    sentences,absolute_ners,absolute_triggers,absolute_relations,absolute_events=json2pot(all_sentences,all_ners,all_relations,all_events,all_sentence_start)
    print('写成pot形式')
    sentences,tags=sen2tag(sentences,absolute_ners,absolute_triggers,absolute_relations,absolute_events)
    print('写入文件')
    write2file(sentences,tags,save_file)

if __name__=='__main__':
    file='./ACE+/english.oneie.json'
    train_file='./json/train.json'
    test_file='./json/test.json'
    dev_file='./json/dev.json'
    print('得到所有句子及类型位置和类型')
    print('生成所有关系的二级标签')
    # sentences,doc_id,sen_id,entities,triggers,relations,events,types=get_index_first(file)
    # sentences,doc_id,sen_id,entities,triggers,relations,events,types=get2new_relation(file)
    sentences,doc_id,sen_id,entities,triggers,relations,events,types=event2pot(file)
    
    # check(sentences,doc_id,sen_id,entities,triggers,relations,events,types)
    # stop
    # sentences,doc_id,sen_id,entities,triggers,relations,events,types=get_index_second(file)
    # sentences,doc_id,sen_id,entities,triggers,relations,events,types=get_index_all_relations(file)

    print('所有的类型',len(list(set(types))))
    print(list(set(types)))
    print('获取划分数据集的doc_id')
    train_doc_id=divide(train_file)
    print('训练集',len(train_doc_id))
    test_doc_id=divide(test_file)
    print('测试集',len(test_doc_id))
    dev_doc_id=divide(dev_file)
    print('验证集',len(dev_doc_id))
    print('划分数据集')
    train_sen,train_tag=index2pot(sentences,doc_id,sen_id,entities,triggers,relations,events,train_doc_id)
    print('训练集句子',len(train_sen))
    test_sen,test_tag=index2pot(sentences,doc_id,sen_id,entities,triggers,relations,events,test_doc_id)
    print('测试集句子',len(test_sen))
    dev_sen,dev_tag=index2pot(sentences,doc_id,sen_id,entities,triggers,relations,events,dev_doc_id)
    print('验证集句子',len(dev_sen))
    print('写入文件')
    write2file(train_sen,train_tag,'./pot_new_second/train.txt')
    write2file(test_sen,test_tag,'./pot_new_second/test.txt')
    write2file(dev_sen,dev_tag,'./pot_new_second/dev.txt')
    print('写入完毕')
    
    # print('json ACE2')
    # train_file='./json/train.json'
    # test_file='./json/test.json'
    # dev_file='./json/dev.json'
    # print('写入训练集')
    # ACE2pot(train_file,'json_new_pot/train.txt')
    # print('写入测试集')
    # ACE2pot(test_file,'json_new_pot/test.txt')
    # print('写入验证集')
    # ACE2pot(dev_file,'json_new_pot/dev.txt')
