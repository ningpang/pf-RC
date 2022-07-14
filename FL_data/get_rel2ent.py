import json

f = open('./tacrev/temp.txt')
rel2ent = {}
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    rel = content[1]
    ent1 = content[2]
    ent2 = content[3]
    if rel == 'no_relation':
        continue

    rel2ent[rel]=[ent1, ent2]
with open('./tacrev/rel2ent.json', 'w') as f:
    json.dump(rel2ent, f)