import json
import os
def get_data(args):
    root_path = 'FL_data/'+args.dataset_name+'/niid_'+args.niid_type
    data_path = root_path+'/'+str(args.num_clients)+'-'+str(args.niid_alpha)
    train_data_list = json.load(open(os.path.join(data_path, 'train.json'), 'r'))
    test_data_list = json.load(open(os.path.join(data_path, 'test.json'), 'r'))
    valid_data = json.load(open(os.path.join(data_path, 'val.json'), 'r'))
    labeled_valid_data = {}
    unlabeled_valid_data = {}
    for relation in valid_data:
        labeled_valid_data[relation] = []
        unlabeled_valid_data[relation] = []
        for i, ins in enumerate(valid_data[relation]):
            if i < args.valid_shot:
                labeled_valid_data[relation].append(ins)
            else:
                unlabeled_valid_data[relation].append(ins)
    return train_data_list, labeled_valid_data, unlabeled_valid_data, test_data_list

def get_relations(args):
    root_path = 'FL_data/' + args.dataset_name
    rel2id = json.load(open(os.path.join(root_path, 'rel2id.json'), 'r'))
    id2rel = {}
    for key in rel2id:
        id2rel[rel2id[key]] = key
    return rel2id, id2rel

def get_entities(args, rel2id, id2rel):
    root_path = 'FL_data/' + args.dataset_name
    rel2ent = json.load(open(os.path.join(root_path, 'rel2ent.json'), 'r'))
    ent2id = {}
    temps = []
    for rel in rel2ent:
        for ent in rel2ent[rel]:
            if ent not in ent2id:
                ent2id[ent] = len(ent2id)
    for id in range(len(id2rel)):
        rel = id2rel[id]
        temps.append([ent2id[rel2ent[rel][0]], rel2id[rel], ent2id[rel2ent[rel][1]]])

    return rel2ent, ent2id, temps