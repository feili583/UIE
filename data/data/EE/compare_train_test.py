'''对比训练集和测试集的实体'''

def read_file(path):
    sentences=[]
    predicts=[]
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i in range(0,len(lines),4):
            sentences.append(lines[i].strip())
            predicts.append(lines[i+2].strip())
    return sentences,predicts

def get_entity(sentences,predicts):
    entity=dict()
    for sen,pred in zip(sentences,predicts):
        preds=pred.split('|')
        sen=sen.split(' ')

        for item in preds:
            if len(item)==0:
                continue
            pos,tag=item.split(' ')
            start,end=pos.split(',')
            start=int(start)
            end=int(end)
            if ' '.join(sen[start:end]) in entity.keys():
                entity[' '.join(sen[start:end])].append(tag)
            else:
                entity[' '.join(sen[start:end])]=[]
                entity[' '.join(sen[start:end])].append(tag)
    return entity

def compare(train_entity,test_entity):
    dif=dict()
    same=dict()
    for key in test_entity.keys():
        if key not in train_entity.keys():
            dif[key]=test_entity[key]
        else:
            same[key]=train_entity[key]+test_entity[key]
    for key in dif.keys():
        print(key,' ',dif[key])
    print()
    print('same')
    for key in same.keys():
        print(key,' ',same[key])

def compare_relation():
    train_file='ACE_relation/pot_train.txt'
    test_file='ACE_relation/pot_test.txt'
    train_sentences,train_predicts=read_file(train_file)
    test_sentences,test_predicts=read_file(test_file)
    train_entity=get_entity(train_sentences,train_predicts)
    test_entity=get_entity(test_sentences,test_predicts)
    compare(train_entity,test_entity)

if __name__ in '__main__':
    train_file='ACE/pot_train_new.txt'
    test_file='ACE/pot_test_new.txt'
    train_sentences,train_predicts=read_file(train_file)
    test_sentences,test_predicts=read_file(test_file)
    train_entity=get_entity(train_sentences,train_predicts)
    test_entity=get_entity(test_sentences,test_predicts)
    compare(train_entity,test_entity)