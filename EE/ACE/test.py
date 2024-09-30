import json
with open('valid_pattern.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data['ner'] + data['event'] + data['role'])

a = [   
            "Instrument",
        "r_Prosecutor",
        "Org",
        "r_Artifact",
        "r_Beneficiary",
        "Place",
        "r_Buyer",
        "Recipient",
        "Origin",
        "r_Adjudicator",
        "Target",
        "r_Agent",
        "r_Vehicle",
        "r_Instrument",
        "Buyer",
        "r_Recipient",
        "Agent",
        "Adjudicator",
        "Artifact",
        "r_Destination",
        "r_Person",
        "Plaintiff",
        "Vehicle",
        "Giver",
        "Attacker",
        "r_Entity",
        "r_Attacker",
        "Person",
        "Entity",
        "r_Org",
        "r_Origin",
        "Destination",
        "Defendant",
        "r_Giver",
        "Victim",
        "r_Plaintiff",
        "r_Victim",
        "Beneficiary",
        "r_Target",
        "r_Place",
        "Seller",
        "Prosecutor",
        "r_Defendant",
        "r_Seller"
]
b = []
for value in a:
    if value.split('.')[0].split('_')[-1] not in b:
        b.append(value.split('.')[0].split('_')[-1])
print(b, len(b), len(a))