import json

from model_tools.get_data_list import get_entities
from model_tools.get_model_list import get_model_list
from model_tools.get_ent_data_loader import get_mem_distill_loader, get_loader, get_con_loader
from .base_frameowrk import Client, Server, RC
from modules import Proto_Net
from modules import Memory_Net
from model_tools.metric_tools import get_acc, get_f1, get_f1_by_relation, get_acc_by_relation
from copy import deepcopy
import torch.optim as optim
from model_tools.get_model_list import get_ente_model
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class PFRCServer(Server):
    def __init__(self, args, clients, unlabeled_val_data, labeled_val_data, rel2id, rel2ent, ent2id, temps):
        super(PFRCServer, self).__init__(args, clients, unlabeled_val_data, rel2id)
        self.labeled_val_data = labeled_val_data
        self.rel2ent = rel2ent
        self.ent2id = ent2id
        self.temps = temps

    def send_state_dict(self):
        weight_list = []
        confidence = self.get_confidence()

        for i in range(self.args.num_clients):
            weight_list.append(self.clients[i].pro_net.prototypes.weight.cpu())
        weight = torch.stack(weight_list, dim=0)
        weight = confidence.unsqueeze(2)*weight
        weight = weight.sum(0)
        for i in range(self.args.num_clients):
            self.clients[i].pro_net.prototypes.weight.data.copy_(weight)


    def get_mem_distill_data(self):
        unlabel_val_loader = get_loader(self.args, self.unlabeled_val_data, rel2id=self.rel2id, rel2ent=self.rel2ent, ent2id=self.ent2id)
        collect_logits = [[] for i in range(self.args.num_clients)]
        collect_tokens = []
        collect_mask = []
        for step, (labels, head, tail, lengthes, tokens, mask) in enumerate(unlabel_val_loader):
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
            collect_tokens.append(tokens.cpu())
            collect_mask.append(mask.cpu())
            for i in range(len(self.clients)):
                with torch.no_grad():
                    self.clients[i].encoder.to(self.args.device)
                    self.clients[i].rel_classifier.to(self.args.device)
                    self.clients[i].ent_classifier.to(self.args.device)
                    self.clients[i].pro_net.to(self.args.device)
                    self.clients[i].mem_net.to(self.args.device)
                    self.clients[i].encoder.eval()
                    self.clients[i].rel_classifier.eval()
                    self.clients[i].ent_classifier.eval()
                    self.clients[i].pro_net.eval()
                    self.clients[i].mem_net.eval()

                    protos = self.clients[i].pro_net.get_all_protos()
                    protos = protos.clone()
                    protos.unsqueeze(0)
                    protos = protos.expand(len(tokens), -1, -1)

                    reps, head, tail = self.clients[i].encoder(tokens, mask)
                    reps = self.clients[i].mem_net(reps, protos)
                    rellogits = self.clients[i].rel_classifier(reps)
                    headlogits = self.clients[i].ent_classifier(head)
                    taillogits = self.clients[i].ent_classifier(tail)
                    res = []
                    for j in self.temps:
                        _res = headlogits[:, j[0]] + rellogits[:, j[1]] + taillogits[:, j[2]]
                        res.append(_res)
                    logits = torch.stack(res, 0).transpose(1, 0)
                    collect_logits[i].append(logits.cpu())
                    # self.clients[i].model.cpu()
        return collect_tokens, collect_mask, collect_logits

    def get_mem_public_results(self):
        unlabel_val_loader = get_loader(self.args, self.unlabeled_val_data, rel2id=self.rel2id, rel2ent=self.rel2ent,
                                        ent2id=self.ent2id)
        collect_logits = [[] for i in range(self.args.num_clients)]
        collect_tokens = []
        collect_mask = []
        collect_labels = []
        for step, (labels, head, tail, lengthes, tokens, mask) in enumerate(unlabel_val_loader):
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
            collect_tokens.append(tokens.cpu())
            collect_mask.append(mask.cpu())
            collect_labels.append(labels.cpu())
            for i in range(len(self.clients)):
                with torch.no_grad():
                    self.clients[i].encoder.to(self.args.device)
                    self.clients[i].rel_classifier.to(self.args.device)
                    self.clients[i].ent_classifier.to(self.args.device)
                    self.clients[i].pro_net.to(self.args.device)
                    self.clients[i].mem_net.to(self.args.device)
                    self.clients[i].encoder.eval()
                    self.clients[i].rel_classifier.eval()
                    self.clients[i].ent_classifier.eval()
                    self.clients[i].pro_net.eval()
                    self.clients[i].mem_net.eval()

                    protos = self.clients[i].pro_net.get_all_protos()
                    protos = protos.clone()
                    protos.unsqueeze(0)
                    protos = protos.expand(len(tokens), -1, -1)

                    reps, head, tail = self.clients[i].encoder(tokens, mask)
                    reps = self.clients[i].mem_net(reps, protos)
                    rellogits = self.clients[i].rel_classifier(reps)
                    headlogits = self.clients[i].ent_classifier(head)
                    taillogits = self.clients[i].ent_classifier(tail)
                    res = []
                    for j in self.temps:
                        _res = headlogits[:, j[0]] + rellogits[:, j[1]] + taillogits[:, j[2]]
                        res.append(_res)
                    logits = torch.stack(res, 0).transpose(1, 0)
                    collect_logits[i].append(logits.cpu())
                    # self.clients[i].model.cpu()
        return collect_labels, collect_tokens, collect_mask, collect_logits

    def get_confidence(self):
        loss_distribution = self.get_client_loss_distribution()
        # data_distribution = self.get_client_data_distribution()
        # confidence_weight = 0.5 * loss_distribution + 0.5 * data_distribution
        confidence_weight = loss_distribution
        return confidence_weight

    def get_distill_loader(self):
        distill_tokens, distill_mask, distill_logits = self.get_mem_distill_data()
        distill_tokens = torch.cat(distill_tokens, 0)
        distill_mask = torch.cat(distill_mask, 0)
        distill_logits = [torch.cat(logit, 0) for logit in distill_logits]
        confidence_weight = self.get_confidence()
        distill_logits = F.softmax(torch.stack(distill_logits, 0), dim=-1)
        weighted_logits = confidence_weight.unsqueeze(1)*distill_logits
        _, vote_label = torch.max(weighted_logits.sum(0), dim=1)
        distill_loader = get_mem_distill_loader(self.args, distill_tokens, distill_mask, vote_label, self.temps)
        return distill_loader

    def get_client_loss_distribution(self):
        distribution = torch.ones((self.args.num_clients, len(self.rel2id)))
        distribution *= -1e7
        for i in range(self.args.num_clients):
            loss_distribution = self.clients[i].get_loss_distribution(self.labeled_val_data)
            distribution[i] = loss_distribution
        distribution = F.softmax(distribution.float(), dim=0)
        return distribution

    # def get_client_data_distribution(self):
    #     distribution = torch.ones((self.args.num_clients, len(self.rel2id)))
    #     distribution *= -1e7
    #     for i in range(self.args.num_clients):
    #         client_distribution = self.clients[i].get_data_distribution()
    #         distribution[i] = client_distribution
    #     distribution = F.softmax(distribution.float(), dim=0)
    #     return distribution

    def load_mem_ckpt(self, load_path=None):
        if load_path is None:
            for i in range(self.args.num_clients):
                path = os.path.join(self.args.pretrain_path, 'client_' + str(i) + '.pt')
                if not os.path.exists(path):
                    print("[Error] The federated mode does not exists !")
                    assert 0
                self.clients[i].model.load_state_dict(torch.load(path))
        else:
            for i in range(self.args.num_clients):
                if not os.path.exists(load_path):
                    print("[Error] The federated mode does not exists !")
                    assert 0
                self.clients[i].encoder.load_state_dict(torch.load(os.path.join(load_path, 'client' + str(i) + '_encoder.pt')))
                self.clients[i].rel_classifier.load_state_dict( torch.load(os.path.join(load_path, 'client' + str(i) + '_rel_classifier.pt')))
                self.clients[i].ent_classifier.load_state_dict(torch.load(os.path.join(load_path, 'client' + str(i) + '_ent_classifier.pt')))
                self.clients[i].mem_net.load_state_dict(torch.load(os.path.join(load_path, 'client' + str(i) + '_mem_net.pt')))
                self.clients[i].pro_net.load_state_dict(torch.load(os.path.join(load_path, 'client' + str(i) + '_pro_net.pt')))

    # def get_test_on_public(self):
    #     collect_labels, distill_tokens, distill_mask, distill_logits = self.get_mem_public_results()
    #     distill_tokens = torch.cat(distill_tokens, 0)
    #     distill_mask = torch.cat(distill_mask, 0)
    #     distill_logits = [torch.cat(logit, 0) for logit in distill_logits]
    #     confidence_weight = self.get_confidence()
    #     distill_logits = F.softmax(torch.stack(distill_logits, 0), dim=-1)
    #     weighted_logits = confidence_weight.unsqueeze(1) * distill_logits
    #     _, vote_label = torch.max(weighted_logits.sum(0), dim=1)
    #     collect_labels = torch.cat(collect_labels, 0)
    #     acc = (collect_labels == vote_label).sum(0) / collect_labels.size(0)
    #     return acc

class PFRCClient(object):
    def __init__(self, args, model_name, train_data, test_data, rel2id, rel2ent, ent2id, temps):
        # super(EntMemoryClient, self).__init__(args, model, train_data, test_data, rel2id)
        self.args = args
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.rel2ent = rel2ent
        self.temps = temps
        self.train_data = train_data
        self.test_data = test_data
        self.model = get_ente_model(model_name, rel2id, ent2id).to(self.args.device)
        self.encoder = self.model.encoder
        self.rel_classifier = self.model.rel_classifier
        self.ent_classifier = self.model.ent_classifier
        self.pro_net = Proto_Net(self.args.input_size, self.args.num_relations).to(self.args.device)
        self.mem_net = Memory_Net(mem_slots=self.args.num_relations,
                                        input_size=self.args.input_size,
                                        output_size=self.args.input_size,
                                        key_size=256,
                                        head_size=self.args.input_size).to(self.args.device)

    def get_data_distribution(self):
        distribution = torch.zeros(len(self.rel2id))
        for relation in self.train_data:
            distribution[self.rel2id[relation]] = len(self.train_data[relation])
        return distribution

    def get_loss_distribution(self, val_data):
        distribution = torch.zeros(len(self.rel2id))
        for relation in val_data:
            cur_data = {}
            cur_data[relation] = val_data[relation]
            loss = self.get_loss(cur_data)
            distribution[self.rel2id[relation]] = -loss
        return distribution

    def get_loss(self, data):
        criterion = nn.CrossEntropyLoss()
        test_data_loader = get_loader(self.args, data, self.rel2id, self.rel2ent, self.ent2id)
        losses = []
        for step, (labels, head, tail, lengthes, tokens, mask) in enumerate(test_data_loader):
            with torch.no_grad():
                self.encoder.to(self.args.device)
                self.rel_classifier.to(self.args.device)
                self.pro_net.to(self.args.device)
                self.mem_net.to(self.args.device)
                self.encoder.eval()
                self.rel_classifier.eval()
                self.pro_net.eval()
                self.mem_net.eval()

                protos = self.pro_net.get_all_protos()
                protos = protos.clone()
                protos.unsqueeze(0)
                protos = protos.expand(len(tokens), -1, -1)

                labels = labels.to(self.args.device)
                head = head.to(self.args.device)
                tail = tail.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)

                reps, head_rep, tail_rep = self.encoder(tokens, mask)
                reps = self.mem_net(reps, protos)
                logits = self.rel_classifier(reps)
                head_logits = self.ent_classifier(head_rep)
                tail_logits = self.ent_classifier(tail_rep)

                loss = criterion(logits, labels)+criterion(head_logits, head)+criterion(tail_logits, tail)
                losses.append(loss.item())
        return np.array(losses).mean()

    def con_train(self):
        self.BertCLS_con_train(self.train_data, self.test_data)

    def BertCLS_con_train(self, train_data, test_data):
        self.encoder.train()
        self.rel_classifier.train()
        self.rel_classifier.train()
        self.pro_net.train()
        criterion = nn.CrossEntropyLoss()
        train_data_loader = get_con_loader(self.args, train_data, self.rel2id, self.rel2ent, self.ent2id)
        optimizer = optim.Adam([{'params': self.encoder.parameters(), 'lr': 0.00001},
                                {'params': self.rel_classifier.parameters(), 'lr': 0.001},
                                {'params': self.ent_classifier.parameters(), 'lr': 0.001},
                                {'params': self.pro_net.parameters(), 'lr': 0.001}])
        for epoch in range(self.args.local_epoch):
            losses = []
            for step, (labels, neg_labels, head, tail, lengthes, tokens, masks) in enumerate(train_data_loader):
                self.encoder.zero_grad()
                self.rel_classifier.zero_grad()
                self.ent_classifier.zero_grad()
                self.pro_net.zero_grad()
                optimizer.zero_grad()

                labels = labels.to(self.args.device)
                head = head.to(self.args.device)
                tail = tail.to(self.args.device)
                neg_labels = torch.stack([x.to(self.args.device) for x in neg_labels], dim=0)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)

                # classification loss
                reps, head_rep, tail_rep = self.encoder(tokens, masks)
                logits = self.rel_classifier(reps)
                head_logits = self.ent_classifier(head_rep)
                tail_logits = self.ent_classifier(tail_rep)
                loss = criterion(logits, labels)+criterion(head_logits, head)+criterion(tail_logits, tail)

                # contrastive loss
                loss_cluster, pos_rep = self.pro_net(reps, labels, neg_labels)

                loss += loss_cluster
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.rel_classifier.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.ent_classifier.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.pro_net.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Contrastive loss is {np.array(losses).mean()}')

    def mem_train(self):
        data_loader = get_loader(self.args, self.train_data, self.rel2id, self.rel2ent, self.ent2id)
        metric, loss = self.BertCLS_mem_train(data_loader, self.test_data)
        return metric, loss

    def BertCLS_mem_train(self, train_data, test_data):
        self.encoder.train()
        self.rel_classifier.train()
        self.ent_classifier.train()
        self.mem_net.train()
        criterion = nn.CrossEntropyLoss()
        # train_data_loader = get_con_loader(self.args, train_data, self.rel2id)
        optimizer = optim.Adam([{'params': self.encoder.parameters(), 'lr': 0.00001},
                                {'params': self.rel_classifier.parameters(), 'lr': 0.001},
                                {'params': self.ent_classifier.parameters(), 'lr': 0.001},
                                {'params': self.mem_net.parameters(), 'lr': 0.001}])

        record_loss = 0.0
        for epoch in range(self.args.local_epoch):
            losses = []
            for step, (labels, head, tail, lengthes, tokens, masks) in enumerate(train_data):
                self.encoder.zero_grad()
                self.rel_classifier.zero_grad()
                self.ent_classifier.zero_grad()
                self.mem_net.zero_grad()
                optimizer.zero_grad()

                protos = self.pro_net.get_all_protos()
                protos = protos.clone()
                protos.unsqueeze(0)
                protos = protos.expand(len(tokens), -1, -1)

                labels = labels.to(self.args.device)
                head = head.to(self.args.device)
                tail = tail.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)

                reps, head_rep, tail_rep = self.encoder(tokens, masks)
                reps = self.mem_net(reps, protos)
                logits = self.rel_classifier(reps)
                head_logits = self.ent_classifier(head_rep)
                tail_loigts = self.ent_classifier(tail_rep)
                loss = criterion(logits, labels)+criterion(head_logits, head)+criterion(tail_loigts, tail)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.rel_classifier.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.ent_classifier.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.mem_net.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Finetuning loss is {np.array(losses).mean()}')
            record_loss = np.array(losses).mean()
        metric = self.Bert_mem_evaluate(test_data)
        return metric, record_loss

    def vote_mem_train(self, data_loader):
        metric, loss = self.BertCLS_mem_train(data_loader, self.test_data)
        return metric, loss

    def Bert_mem_evaluate(self, test_data):
        self.encoder.eval()
        self.rel_classifier.eval()
        self.ent_classifier.eval()
        self.mem_net.eval()
        grnd = []
        pred = []
        metric = 0
        test_data_loader = get_con_loader(self.args, test_data, self.rel2id, self.rel2ent, self.ent2id, batch_size=1)
        for step, (labels, neg_labels, head, tail, lengthes, tokens, mask) in enumerate(test_data_loader):
            with torch.no_grad():
                protos = self.pro_net.get_all_protos()
                protos = protos.clone()
                protos.unsqueeze(0)
                protos = protos.expand(len(tokens), -1, -1)

                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in mask], dim=0)

                reps, head_rep, tail_rep = self.encoder(tokens, masks)
                reps = self.mem_net(reps, protos)
                rellogits = self.rel_classifier(reps)
                headlogits = self.ent_classifier(head_rep)
                taillogits = self.ent_classifier(tail_rep)
                res = []
                for i in self.temps:
                    _res = headlogits[:, i[0]] + rellogits[:, i[1]] + taillogits[:, i[2]]
                    res.append(_res)
                logits = torch.stack(res, 0).transpose(1, 0)
                max_id = np.argmax(logits.cpu().data.numpy(), axis=1).tolist()
                labels = labels.view(-1).detach().cpu().tolist()
                grnd += labels
                pred += max_id
        if self.args.metric == 'Acc':
            metric = get_acc(grnd, pred)
        elif self.args.metric == 'F1':
            metric = get_f1(grnd, pred, 0)
        return metric
    # def quantitive_evaluate(self, test_data):
    #     self.encoder.eval()
    #     self.rel_classifier.eval()
    #     self.ent_classifier.eval()
    #     self.mem_net.eval()
    #     grnd = []
    #     pred = []
    #     metric = 0
    #     test_data_loader = get_con_loader(self.args, test_data, self.rel2id, self.rel2ent, self.ent2id, batch_size=1)
    #     for step, (labels, neg_labels, head, tail, lengthes, tokens, mask) in enumerate(test_data_loader):
    #         with torch.no_grad():
    #             protos = self.pro_net.get_all_protos()
    #             protos = protos.clone()
    #             protos.unsqueeze(0)
    #             protos = protos.expand(len(tokens), -1, -1)
    #
    #             labels = labels.to(self.args.device)
    #             tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
    #             masks = torch.stack([x.to(self.args.device) for x in mask], dim=0)
    #
    #             reps, head_rep, tail_rep = self.encoder(tokens, masks)
    #             reps = self.mem_net(reps, protos)
    #             rellogits = self.rel_classifier(reps)
    #             headlogits = self.ent_classifier(head_rep)
    #             taillogits = self.ent_classifier(tail_rep)
    #             res = []
    #             for i in self.temps:
    #                 _res = headlogits[:, i[0]] + rellogits[:, i[1]] + taillogits[:, i[2]]
    #                 res.append(_res)
    #             logits = torch.stack(res, 0).transpose(1, 0)
    #             max_id = np.argmax(logits.cpu().data.numpy(), axis=1).tolist()
    #             labels = labels.view(-1).detach().cpu().tolist()
    #             grnd += labels
    #             pred += max_id
    #     metric = get_acc(grnd, pred)
    #     return metric, grnd, pred

    def save_mem_ckpt(self, id, save_path=None):
        if save_path is None:
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)
            torch.save(self.encoder.state_dict(), os.path.join(self.args.save_path, 'client' + str(id) + '_encoder.pt'))
            torch.save(self.rel_classifier.state_dict(), os.path.join(self.args.save_path, 'client' + str(id) + '_rel_classifier.pt'))
            torch.save(self.ent_classifier.state_dict(), os.path.join(self.args.save_path, 'client' + str(id) + '_ent_classifier.pt'))
            torch.save(self.mem_net.state_dict(), os.path.join(self.args.save_path, 'client' + str(id) + '_mem_net.pt'))
            torch.save(self.pro_net.state_dict(), os.path.join(self.args.save_path, 'client' + str(id) + '_pro_net.pt'))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.encoder.state_dict(), os.path.join(save_path, 'client' + str(id) + '_encoder.pt'))
            torch.save(self.rel_classifier.state_dict(), os.path.join(save_path, 'client' + str(id) + '_rel_classifier.pt'))
            torch.save(self.ent_classifier.state_dict(), os.path.join(save_path, 'client' + str(id) + '_ent_classifier.pt'))
            torch.save(self.mem_net.state_dict(), os.path.join(save_path, 'client' + str(id) + '_mem_net.pt'))
            torch.save(self.pro_net.state_dict(), os.path.join(save_path, 'client' + str(id) + '_pro_net.pt'))

class PFRC(RC):
    def __init__(self, args):
        super(PFRC, self).__init__(args)
        self.rel2ent, self.ent2id, self.temps = get_entities(args, self.rel2id, self.id2rel)

        self.clients = [
            PFRCClient(args, self.model_list[i], self.train_data_list[i], self.test_data_lits[i], self.rel2id, self.rel2ent, self.ent2id, self.temps)
            for i in range(args.num_clients)]
        self.server = PFRCServer(args, self.clients, self.unlabeled_valid_data, self.labeled_valid_data, self.rel2id, self.rel2ent, self.ent2id, self.temps)

    def train(self):

        bad_count = 0
        distill_performance = {idx: 0 for idx in range(self.num_client)}
        local_performance = {idx: 0 for idx in range(self.num_client)}
        dist_best_performance = {idx: 0 for idx in range(self.num_client)}
        local_best_performance = {idx: 0 for idx in range(self.num_client)}

        if self.args.load_ckpt:
            self.server.load_mem_ckpt(self.args.pretrain_path)
            print('Successfully load the Checkpoint !!!')
        else:
            print('There are no pretrained models ...')
            print('----Starts Pretraining----')
            for epoch in range(10):
                print('epoch:', epoch)
                sample_set = range(self.args.num_clients)
                for i in sample_set:
                    print(f'----Client {i} Starts Pretraining----')
                    self.clients[i].con_train()
                    cur_metric, loss = self.clients[i].mem_train()
                    print(f'Client {i} Acc: {cur_metric}')
                    if cur_metric > local_performance[i]:
                        local_performance[i] = cur_metric
                        self.clients[i].save_mem_ckpt(i, self.args.pretrain_path)

        print('----Distillation for Clients----')
        print()
        last_distill = 0.0
        last_local = 0.0
        for epoch in range(self.args.global_epoch):
            # n_sample = max(round(self.args.fraction*self.num_client), 1)
            # sample_set = np.random.choice(self.num_client, n_sample, replace=False)
            distill_loader = self.server.get_distill_loader()

            self.server.send_state_dict()
            print('----Got the Distill Data----')

            sample_set = range(self.args.num_clients)
            print(f"Global Epoch: {epoch}, Clients Include: {sample_set}")
            print()
            for i in iter(sample_set):

                print(f'----Client {i} Starts Local Con Training----')
                self.clients[i].con_train()
                print(f'----Client {i} Starts Distill Training----')
                cur_metric, loss = self.clients[i].vote_mem_train(deepcopy(distill_loader))
                distill_performance[i] = cur_metric
                print(f'Client {i} Acc: {cur_metric}')
                if cur_metric > dist_best_performance[i]:
                    dist_best_performance[i] = cur_metric
                    # self.clients[i].save_mem_ckpt(i)
                print(f'----Client {i} Starts Local Mem Training----')
                cur_metric, loss = self.clients[i].mem_train()
                local_performance[i] = cur_metric
                print(f'Client {i} Acc: {cur_metric}')
                if cur_metric > local_best_performance[i]:
                    local_best_performance[i] = cur_metric
                    self.clients[i].save_mem_ckpt(i)
                print()
            for i in range(self.args.num_clients):
                self.add_dist_performace(i, distill_performance[i], epoch)
                self.add_local_performance(i, local_performance[i], epoch)

            distill_best_mean = np.mean(list(dist_best_performance.values()))
            distill_best_std = np.std(list(dist_best_performance.values()))
            local_best_mean = np.mean(list(local_best_performance.values()))
            locall_best_std = np.std(list(local_best_performance.values()))
            self.add_best_distill(distill_best_mean, distill_best_std, epoch)
            self.add_best_local(local_best_mean, locall_best_std, epoch)
            if distill_best_mean > last_distill or local_best_mean > last_local:
                bad_count = 0
            else:
                bad_count += 1
            marker = ''
            if distill_best_mean > last_distill:
                marker = '*'
                last_distill = distill_best_mean
            print(f'Distill Performance: {dist_best_performance}')
            print(f'Mean:{distill_best_mean}, Std:{distill_best_std}' + marker)
            marker = ''
            if local_best_mean > last_local:
                marker = '*'
                last_local = local_best_mean
            print(f'Local Performance: {local_best_performance}')
            print(f'Mean:{local_best_mean}, Std:{locall_best_std}' + marker)

            print(f'Bad count: {bad_count} !')
            print()

            if bad_count > self.args.early_stop:
                print('Early Stop !!!!')
                break

    # def test(self):
    #     self.server.load_mem_ckpt(self.args.save_path)
    #     sample_set = range(self.args.num_clients)
    #     all_pred = []
    #     all_grnd = []
    #     avg_acc = 0
    #     for i in iter(sample_set):
    #         metric, grnd, pred = self.clients[i].quantitive_evaluate(self.clients[i].test_data)
    #         all_grnd += grnd
    #         all_pred += pred
    #         avg_acc += metric
    #     avg_acc/=self.args.num_clients
    #     metric = get_f1_by_relation(all_grnd, all_pred, self.rel2id)
    #     print(avg_acc)
    #     return metric
    #
    # def test_on_public(self):
    #     self.server.load_mem_ckpt(self.args.save_path)
    #     metric = self.server.get_test_on_public()
    #     return  metric
    #
    # def case_test(self, data):
    #     self.server.load_mem_ckpt(self.args.save_path)
    #     results = {}
    #     for relation in data:
    #         results[relation] = []
    #         for instance in data[relation]:
    #             current_instance = {}
    #             current_instance[relation] = [instance[0]]
    #             data_loader = get_loader(self.args, current_instance, rel2id=self.rel2id, rel2ent=self.rel2ent,
    #                                     ent2id=self.ent2id)
    #             for step, (labels, head, tail, lengthes, tokens, mask) in enumerate(data_loader):
    #                 tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
    #                 mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
    #                 collect_logits = [[] for i in range(self.args.num_clients)]
    #                 for i in range(len(self.clients)):
    #                     with torch.no_grad():
    #                         self.clients[i].encoder.to(self.args.device)
    #                         self.clients[i].rel_classifier.to(self.args.device)
    #                         self.clients[i].ent_classifier.to(self.args.device)
    #                         self.clients[i].pro_net.to(self.args.device)
    #                         self.clients[i].mem_net.to(self.args.device)
    #                         self.clients[i].encoder.eval()
    #                         self.clients[i].rel_classifier.eval()
    #                         self.clients[i].ent_classifier.eval()
    #                         self.clients[i].pro_net.eval()
    #                         self.clients[i].mem_net.eval()
    #
    #                         protos = self.clients[i].pro_net.get_all_protos()
    #                         protos = protos.clone()
    #                         protos.unsqueeze(0)
    #                         protos = protos.expand(len(tokens), -1, -1)
    #
    #                         reps, head, tail = self.clients[i].encoder(tokens, mask)
    #                         reps = self.clients[i].mem_net(reps, protos)
    #                         rellogits = self.clients[i].rel_classifier(reps)
    #                         headlogits = self.clients[i].ent_classifier(head)
    #                         taillogits = self.clients[i].ent_classifier(tail)
    #                         res = []
    #                         for j in self.temps:
    #                             _res = headlogits[:, j[0]] + rellogits[:, j[1]] + taillogits[:, j[2]]
    #                             res.append(_res)
    #                         logits = torch.stack(res, 0).transpose(1, 0)
    #                         collect_logits[i].append(logits.cpu())
    #                 distill_logits = [torch.cat(logit, 0) for logit in collect_logits]
    #                 # weighted_logits = torch.cat(distill_logits, 0).mean(0)
    #                 # _, vote_label = torch.max(weighted_logits, dim=0)
    #                 confidence_weight = self.server.get_confidence()
    #                 distill_logits = F.softmax(torch.stack(distill_logits, 0), dim=-1)
    #                 weighted_logits = confidence_weight.unsqueeze(1) * distill_logits
    #                 _, vote_label = torch.max(weighted_logits.sum(0), dim=1)
    #                 if vote_label.item() == labels.item():
    #                     print(relation)
    #                     instance.append(vote_label.cpu().numpy().tolist())
    #                     results[relation].append(instance)
    #                     # print('correct')
    #                     # continue
    #                 else:
    #                     # print(relation)
    #                     # instance.append(weighted_logits.cpu().numpy().tolist())
    #                     # results[relation].append(instance)
    #                     continue
    #     return results