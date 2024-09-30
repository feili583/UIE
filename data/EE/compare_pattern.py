import json

if __name__=='__main__':
    with open('../../../OneIE-main/resource/valid_patterns/event_role.json') as f:
        event_role_one=json.load(f)
    with open('../../../OneIE-main/resource/valid_patterns/relation_entity.json') as f:
        relation_entity_one=json.load(f)
    with open('../../../OneIE-main/resource/valid_patterns/role_entity.json') as f:
        role_entity_one=json.load(f)
    with open('./ACE/valid_pattern.json') as f:
        ace=json.load(f)
    with open('./ACE_add/valid_pattern.json') as f:
        ace_add=json.load(f)

    print(len(event_role_one),len(ace['event_role']),len(ace_add['event_role']))
    print(len(set(event_role_one.keys())&set(ace['event_role'].keys())),len(set(event_role_one)&set(ace_add['event_role'])))
    print(len(relation_entity_one),len(ace['relation_entity']),len(ace_add['relation_entity']))
    print(len(set(relation_entity_one.keys())&set(ace['relation_entity'].keys())),len(set(relation_entity_one)&set(ace_add['relation_entity'])))
    print(len(role_entity_one),len(ace['role_entity']),len(ace_add['role_entity']))
    print(len(set(role_entity_one)&set(ace['role_entity'])),len(set(role_entity_one)&set(ace_add['role_entity'])))
    