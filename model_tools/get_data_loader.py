import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=['[E11]', '[E12]', '[E21]', '[E22]'])

def process(args, data, rel2id):
    processed_data = []
    for relation in data:
        for ins in data[relation]:
            tokenized_sample = {}
            tokenized_sample['relation'] = rel2id[ins['relation']]
            h1, h2 = ins['h']['pos'][0], ins['h']['pos'][1]
            t1, t2 = ins['t']['pos'][0], ins['t']['pos'][1]

            tokens = []
            for idx, token in enumerate(ins['token']):
                if idx == t2:
                    tokens.append('[E22]')
                if idx == h1:
                    tokens.append('[E11]')
                if idx == h2:
                    tokens.append('[E12]')
                if idx == t1:
                    tokens.append('[E21]')
                tokens.append(token)

            tokenized_sample['tokens'] = tokenizer.encode(' '.join(tokens),
                                                          padding='max_length',
                                                          truncation=True,
                                                          max_length=args.max_length)

            e11 = np.argwhere(np.array(tokenized_sample['tokens']) == 30522)
            e21 = np.argwhere(np.array(tokenized_sample['tokens']) == 30524)
            if len(e11)<1 or len(e21)< 1:
                continue
            # print(e11, e21)
            length = np.argwhere(np.array(tokenized_sample['tokens']) == 102)[0][0]
            mask = [1] * length + [0] * (args.max_length - length)

            tokenized_sample['length'] = length
            tokenized_sample['mask'] = mask
            processed_data.append(tokenized_sample)

    return processed_data

def con_process(args, data, rel2id):
    processed_data = []
    for relation in data:
        for ins in data[relation]:
            tokenized_sample = {}
            tokenized_sample['relation'] = rel2id[ins['relation']]
            tokenized_sample['neg_relation'] = [rel2id[rel] for rel in rel2id if rel != ins['relation']]
            h1, h2 = ins['h']['pos'][0], ins['h']['pos'][1]
            t1, t2 = ins['t']['pos'][0], ins['t']['pos'][1]

            tokens = []
            for idx, token in enumerate(ins['token']):
                if idx == t2:
                    tokens.append('[E22]')
                if idx == h1:
                    tokens.append('[E11]')
                if idx == h2:
                    tokens.append('[E12]')
                if idx == t1:
                    tokens.append('[E21]')
                tokens.append(token)

            tokenized_sample['tokens'] = tokenizer.encode(' '.join(tokens),
                                                          padding='max_length',
                                                          truncation=True,
                                                          max_length=args.max_length)

            e11 = np.argwhere(np.array(tokenized_sample['tokens']) == 30522)
            e21 = np.argwhere(np.array(tokenized_sample['tokens']) == 30524)
            if len(e11)<1 or len(e21)< 1:
                continue
            # print(e11, e21)
            length = np.argwhere(np.array(tokenized_sample['tokens']) == 102)[0][0]
            mask = [1] * length + [0] * (args.max_length - length)

            tokenized_sample['length'] = length
            tokenized_sample['mask'] = mask
            processed_data.append(tokenized_sample)

    return processed_data


class data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        length = torch.tensor([item['length'] for item in data])
        tokens = [torch.tensor(item['tokens']) for item in data]
        masks = [torch.tensor(item['mask']) for item in data]
        return (label, length, tokens, masks)

def get_data_loader(args, data, shuffle=True, drop_last = False, batch_size=None):
    dataset = data_set(data, args)

    if batch_size == None:
        batch_size = min(args.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )
    return data_loader

def get_loader(args, data, rel2id, batch_size=None):
    data = process(args, data, rel2id)
    loader = get_data_loader(args, data, batch_size=batch_size)
    return loader

class con_data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        neg_label = [torch.tensor(item['neg_relation']) for item in data]
        length = torch.tensor([item['length'] for item in data])
        tokens = [torch.tensor(item['tokens']) for item in data]
        masks = [torch.tensor(item['mask']) for item in data]
        return (label, neg_label, length, tokens, masks)

def get_con_data_loader(args, data, shuffle=True, drop_last = False, batch_size=None):
    dataset = con_data_set(data, args)

    if batch_size == None:
        batch_size = min(args.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )
    return data_loader

def get_con_loader(args, data, rel2id, batch_size=None):
    data = con_process(args, data, rel2id)
    loader = get_con_data_loader(args, data, batch_size=batch_size)
    return loader

class MyDataset(Dataset):
    def __init__(self, tokens, mask, logits):
        self.len = tokens.size(0)
        self.tokens = tokens
        self.mask = mask
        self.logits = logits

    def __getitem__(self, item):
        return self.tokens[item], self.mask[item], self.logits[item]

    def __len__(self):
        return self.len

def get_distill_loader(args, tokens, mask, logits, shuffle=True, drop_last = False, batch_size=None):
    dataset = MyDataset(tokens, mask, logits)
    if batch_size == None:
        batch_size = min(args.batch_size, len(dataset))
    else:
        batch_size = min(batch_size, len(dataset))

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=False,
                              num_workers=args.num_workers,
                              drop_last=drop_last)
    return train_loader

class MyConDataset(Dataset):
    def __init__(self, tokens, mask, logits):
        self.len = tokens.size(0)
        self.tokens = tokens
        self.mask = mask
        self.logits = logits

    def __getitem__(self, item):
        return self.logits[item], 0, self.tokens[item], self.mask[item],

    def __len__(self):
        return self.len

def get_mem_distill_loader(args, tokens, mask, logits, shuffle=True, drop_last = False, batch_size=None):
    dataset = MyConDataset(tokens, mask, logits)
    if batch_size == None:
        batch_size = min(args.batch_size, len(dataset))
    else:
        batch_size = min(batch_size, len(dataset))

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=False,
                              num_workers=args.num_workers,
                              drop_last=drop_last)
    return train_loader