import json
#检查一下边界问题
##11.17评估如果出现预测在相同的实体之间需要去重，否则会出现预测多个相同的关系
#11.18评估出现多个相同关系说明召回多是对的不需要去重，加入只使用oneie进行评估，golden数量使用原始和oneie构造的训练文件

def read_gold_file(predict_file,valid_pattern_path):
    predict_res_dict_list = []
    sen_list = []
    with open(valid_pattern_path,'r',encoding='utf-8') as f:
        valid_pattern = json.load(f)
    with open(predict_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for index in range(0,len(lines),4):
            sen = lines[index].strip()
            predict_res_dict = {}
            predict_dict = {}
            predict_dict['ner'] = []
            predict_dict['ner_id'] = []
            predict_dict['trigger'] = []
            predict_dict['trigger_id'] = []
            predict_dict['relation'] = []
            predict_dict['role'] = []
            predict_dict['role_id'] = []
            if len(lines[index + 2].strip())==0:
                sen_list.append(sen)
                predict_res_dict[sen] = predict_dict
                predict_res_dict_list.append(predict_res_dict)
                continue
            for index_types in lines[index + 2].strip().split('|'):
                index_,types = index_types.split(' ')
                start,end=index_.split(',')
                start = int(start)
                end = int(end) - 1
                if types in valid_pattern['ner']:
                    predict_dict['ner'].append([start,end,types])
                    predict_dict['ner_id'].append([start,end])
                if types in valid_pattern['event']:
                    predict_dict['trigger'].append([start,end,types])
                    predict_dict['trigger_id'].append([start,end])
                if types in valid_pattern['relation']:
                    predict_dict['relation'].append([start,end,types])
                if types in valid_pattern['role']:
                    predict_dict['role'].append([start,end,types])
            predict_dict_list = []
            for relation_ in predict_dict['relation']:
                tmp_relation = []
                start,end,types = relation_
                for ind_ in range(start,end):
                    if [start,ind_] in predict_dict['ner_id']:
                        tmp_relation.extend([(start),(ind_)])
                        break
                for ind_ in range(start+1,end+1):
                    if [ind_,end] in predict_dict['ner_id']:
                        tmp_relation.extend([(ind_),(end)])
                        break
                if len(tmp_relation)==4:
                    tmp_relation.extend([types])
                    predict_dict_list.append(tmp_relation)
            predict_dict['relation'] = predict_dict_list
            predict_dict_list = []
            predict_dict_id_list = []
            for role_ in predict_dict['role']:
                tmp_role =[]
                tmp_role_id = []
                start,end,types = role_
                for ind_ in range(start,end):
                    if [start,ind_] in predict_dict['ner_id']:
                        tmp_role.extend([(start),(ind_)])
                        tmp_role_id.extend([(start),(ind_)])
                        break
                if len(tmp_role) > 0:
                    for ind_ in range(start+1,end+1):
                        if [ind_,end] in predict_dict['trigger_id']:
                            tmp_role.extend([(ind_),(end)])
                            tmp_role_id.extend([(ind_),(end)])
                            break
                else:
                    for ind_ in range(start,end):
                        if [start,ind_] in predict_dict['trigger_id']:
                            tmp_role.extend([(start),(ind_)])
                            tmp_role_id.extend([(start),(ind_)])
                            break
                    for ind_ in range(start+1,end+1):
                        if [ind_, end] in predict_dict['ner_id']:
                            tmp_role.extend([(ind_),(end)])
                            tmp_role_id.extend([(ind_),(end)])

                tmp_role.extend([types])
               
                if tmp_role not in predict_dict_list:
                    predict_dict_list.append(tmp_role)
                    predict_dict_id_list.append(tmp_role_id)
                predict_dict['role'] = predict_dict_list
                predict_dict['role_id'] = predict_dict_id_list
            predict_res_dict[sen] = predict_dict
            predict_res_dict_list.append(predict_res_dict)
            sen_list.append(sen)
    with open('test_gold.json','w',encoding = 'utf-8') as f:
        f.write(json.dumps(predict_res_dict_list,indent = 4))
    return predict_res_dict_list, sen_list

def read_predict_file(predict_file,valid_pattern_path,sen_list):
    predict_res_dict_list = []
    with open(valid_pattern_path,'r',encoding='utf-8') as f:
        valid_pattern = json.load(f)
    with open(predict_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        sen_index = 0
        for index in range(0,len(lines),3):
            # sen = lines[index].strip()
            sen = sen_list[sen_index]
            sen_index += 1
            predict_dict = {}
            predict_res_dict = {}
            predict_dict['ner'] = []
            predict_dict['trigger'] = []
            predict_dict['relation'] = []
            predict_dict['role'] = []
            predict_dict['trigger_id'] = []
            predict_dict['role_id'] = []
            if len(lines[index + 1].strip())==0:
                predict_res_dict[sen] = predict_dict
                predict_res_dict_list.append(predict_res_dict)
                continue
            
            for index_types in lines[index + 1].strip().split('|'):
                start,end,types = index_types.split(',')
                # start,end=index.split(',')
                start = int(start)
                end = int(end)
                if types in valid_pattern['ner']:
                    predict_dict['ner'].append([start,end,types])
                if types in valid_pattern['event']:
                    predict_dict['trigger'].append([start,end,types])
                    predict_dict['trigger_id'].append([start,end])
                if types in valid_pattern['relation']:
                    predict_dict['relation'].append([start,end,types])
                if types in valid_pattern['role']:
                    predict_dict['role'].append([start,end,types])
            predict_dict_list = []
            for relation_ in predict_dict['relation']:
                tmp_relation = []
                start,end,types = relation_
                for ner_start,ner_end,_ in predict_dict['ner']:
                    if start >= ner_start and start <= ner_end and (end < ner_start or end > ner_end):
                        tmp_relation.extend([ner_start,ner_end])
                        break
                for ner_start,ner_end,_ in predict_dict['ner']:
                    if end >= ner_start and end <= ner_end and (start < ner_start or start > ner_end):
                        tmp_relation.extend([ner_start,ner_end])
                        break
                tmp_relation.extend([types])
                predict_dict_list.append(tmp_relation)
            predict_dict['relation'] = predict_dict_list
            predict_dict_list = []
            predict_id_dict_list = []

            for role_ in predict_dict['role']:
                tmp_role = []
                tmp_role_id = []
                start,end,types = role_
                for ner_start,ner_end,_ in predict_dict['ner']:
                    if start >= ner_start and start <= ner_end and (end < ner_start or end > ner_end):
                        tmp_role.extend([ner_start,ner_end])
                        tmp_role_id.extend([ner_start,ner_end])
                        break
                if len(tmp_role) > 0:
                    for ner_start,ner_end,_ in predict_dict['trigger']:
                        if end >= ner_start and end <= ner_end and (start < ner_start or start > ner_start):
                            tmp_role.extend([ner_start,ner_end])
                            tmp_role_id.extend([ner_start,ner_end])
                            break
                else:
                    for ner_start,ner_end,_ in predict_dict['trigger']:
                        if start >= ner_start and start <= ner_end and (end < ner_start or end > ner_end):
                            tmp_role.extend([ner_start,ner_end])
                            tmp_role_id.extend([ner_start,ner_end])
                            break
                    for ner_start,ner_end,_ in predict_dict['ner']:
                        if end >= ner_start and end <= ner_end and (start < ner_start or start > ner_start):
                            tmp_role.extend([ner_start,ner_end])
                            tmp_role_id.extend([ner_start,ner_end])
                            break
                if len(tmp_role) == 4:
                    predict_id_dict_list.append(tmp_role_id)
                    tmp_role.extend([types])
                    predict_dict_list.append(tmp_role)

            predict_dict['role'] = predict_dict_list
            predict_dict['role_id'] = predict_id_dict_list
            predict_res_dict[sen] = predict_dict
            predict_res_dict_list.append(predict_res_dict)
    with open('test_predict.json','w',encoding = 'utf-8') as f:
        f.write(json.dumps(predict_res_dict_list,indent = 4))
    return predict_res_dict_list

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def score_res(predict_dict, gold_dict, ner_count, trigger_count, relation_count, role_count):
    pred_count = 0
    gold_count = 0
    gold_arg_num = pred_arg_num = 0
    arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = 0
    trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0
    # print(predict_dict)
    # print(gold_dict)
    # print(gold_dict_all)
    for index in range(len(gold_dict)):
        keys = [key for key in gold_dict[index].keys()]
        key = keys[0]
        gold_entities = gold_dict[index][key]['ner']
        # gold_entities_all = gold_dict_all[index][key]['ner']
        if index < len(predict_dict):
            pred_entities = predict_dict[index][key]['ner']
        # gold_ent_num += len(gold_entities_all)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities if entity in gold_entities])

        gold_relations = gold_dict[index][key]['relation']
        # gold_relations_all = gold_dict_all[index][key]['relation']
        if index < len(predict_dict):
            pred_relations = predict_dict[index][key]['relation']
        # gold_rel_num += len(gold_relations_all)
        pred_rel_num += len(pred_relations)
        rel_match_num += len([relation for relation in pred_relations if relation in gold_relations])

        gold_triggers = gold_dict[index][key]['trigger']
        # gold_triggers_all = gold_dict_all[index][key]['trigger']
        if index < len(predict_dict):
            pred_triggers = predict_dict[index][key]['trigger']
        # gold_trigger_num += len(gold_triggers_all)
        pred_trigger_num += len(pred_triggers)
        for trg_start, trg_end, event_type in pred_triggers:
            matched = [item for item in gold_triggers if item[0] == trg_start and item[1] == trg_end]
            if matched:
                trigger_idn_num += 1
                if matched[0][-1] == event_type:
                    trigger_class_num += 1

        gold_args = gold_dict[index][key]['role']
        # gold_args_all = gold_dict_all[index][key]['role']
        if index < len(predict_dict):
            pred_args = predict_dict[index][key]['role']

        # gold_arg_num += len(gold_args_all)
        pred_arg_num += len(pred_args)
        for pred_arg in pred_args:
            arg_start, arg_end, arg_start_, arg_end_, role = pred_arg
            gold_idn = [item for item in gold_args
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == arg_start_ and item[3] == arg_end_]
            if gold_idn:
                arg_idn_num += 1
                gold_class = [item for item in gold_idn if item[-1] == role]
                if gold_class:
                    arg_class_num += 1
    
    gold_ent_num = ner_count
    gold_trigger_num = trigger_count
    gold_arg_num = role_count
    gold_rel_num = relation_count

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_idn_num)
    trigger_prec, trigger_rec, trigger_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_class_num)
    relation_prec, relation_rec, relation_f = compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num)
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print('Entity: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0))
    print('Trigger identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
    print('Trigger: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
    print('Relation: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        relation_prec * 100.0, relation_rec * 100.0, relation_f * 100.0))
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
        'trigger': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
        'trigger_id': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
                       'f': trigger_id_f},
        'role': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
        'role_id': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
        'relation': {'prec': relation_prec, 'rec': relation_rec,
                     'f': relation_f}
    }
    return scores


def transfer_all2part(gold_dict_all,sen_list):
    gold_dict_part = []
    print(len(gold_dict_all),len(sen_list))
    for gold_dict in gold_dict_all:
        for sen in sen_list:
            keys = [key for key in gold_dict.keys()]
            key = keys[0]
            if sen in key:
                tmp = {}
                tmp[sen] = gold_dict[key]
                gold_dict_part.append(tmp)
                break
    print(len(gold_dict_part),len(gold_dict_all))
    return gold_dict_part

def count_gold(data_dict_list):
    ner = 0
    trigger = 0
    relation = 0
    role = 0
    for data_dict in data_dict_list:
        for key in data_dict.keys():
            ner += len(data_dict[key]['ner'])
            trigger += len(data_dict[key]['trigger'])
            relation += len(data_dict[key]['relation'])
            role += len(data_dict[key]['role'])
    return ner, trigger, relation, role


if __name__=='__main__':
    # print('train')
    # predict_path1 = './predict_file/EE/ee.txttrain_eval.txt'
    # gold_path2 = './data/EE/ACE/pot_train_oneie.txt'
    # gold_path3 = './data/EE/ACE/pot_train_oneie_infer.txt'
    # valid_pattern_path = './data/EE/ACE/valid_pattern.json'
    # gold_dict_all, sen_list = read_gold_file(gold_path2,valid_pattern_path)
    # gold_dict, sen_list = read_gold_file(gold_path3,valid_pattern_path)
    # # gold_dict_all = transfer_all2part(gold_dict_all,sen_list)
    # predict_dict = read_predict_file(predict_path1,valid_pattern_path,sen_list)
    # ner_count, trigger_count, relation_count, role_count = count_gold(gold_dict_all)
    # score_res(predict_dict, gold_dict, ner_count, trigger_count, relation_count, role_count)
    print('dev')
    predict_path1 = './predict_file/EE/ee.txtdev_eval.txt'
    gold_path2 = './data/EE/ACE/pot_dev_oneie.txt'
    gold_path3 = './data/EE/ACE/pot_dev_oneie_infer.txt'
    valid_pattern_path = './data/EE/ACE/valid_pattern.json'
    gold_dict_all, sen_list = read_gold_file(gold_path2,valid_pattern_path)
    gold_dict, sen_list = read_gold_file(gold_path3,valid_pattern_path)
    # gold_dict_all = transfer_all2part(gold_dict_all,sen_list)
    predict_dict = read_predict_file(predict_path1,valid_pattern_path,sen_list)
    ner_count, trigger_count, relation_count, role_count = count_gold(gold_dict_all)
    score_res(predict_dict, gold_dict, ner_count, trigger_count, relation_count, role_count)

    print('test')
    predict_path1 = './predict_file/EE/ee.txttest_eval.txt'
    gold_path2 = './data/EE/ACE/pot_test_oneie.txt'
    gold_path3 = './data/EE/ACE/pot_test_oneie_infer.txt'
    valid_pattern_path = './data/EE/ACE/valid_pattern.json'
    gold_dict_all, sen_list = read_gold_file(gold_path2,valid_pattern_path)
    gold_dict, sen_list = read_gold_file(gold_path3,valid_pattern_path)
    # gold_dict_all = transfer_all2part(gold_dict_all,sen_list)
    predict_dict = read_predict_file(predict_path1,valid_pattern_path,sen_list)
    ner_count, trigger_count, relation_count, role_count = count_gold(gold_dict_all)
    score_res(predict_dict, gold_dict, ner_count, trigger_count, relation_count, role_count)
