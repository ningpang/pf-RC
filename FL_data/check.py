import json
# data = json.load(open('./tacred/niid_quantity/8-100/train.json', 'r'))
# rel2id = json.load(open('./tacred/rel2id.json', 'r'))
# num = 0
# for r in data[0]:
#     print(num, len(data[0][r]))
#     num += 1

relname = json.load(open('./tacred/rel2id.json', 'r'))
# names = json.load(open('./tacred/relnames_tacred.json', 'r'))
file = open('./tacred/temp.txt', 'r')
temps = file.readlines()
names = {}
for temp in temps:
    line = temp.strip().split()
    name = line[1]
    head_type = line[2]
    tail_type = line[6]
    if name not in names:
        names[name] = [head_type, tail_type]
with open('tacred/rel2ent.json', 'w') as f:
    json.dump(names, f)
print(len(relname), len(names))
for key in relname:
    print(names[key])