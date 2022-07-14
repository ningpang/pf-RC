import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model_tools.get_data_loader import get_loader
from model_tools.metric_tools import get_acc, get_f1
from pytorch_pretrained_bert.optimization import BertAdam
from model_tools.get_data_list import get_data, get_relations
from model_tools.get_model_list import get_model, get_model_list
from model_tools.get_data_loader import get_mem_distill_loader, get_loader, get_con_loader
class Server(object):
    def __init__(self, args, clients, unlabeled_val_data, rel2id):
        self.args = args
        self.clients = clients
        self.unlabeled_val_data = unlabeled_val_data
        self.rel2id = rel2id

    def get_distill_data(self):
        unlabel_val_loader = get_loader(self.args, self.unlabeled_val_data, rel2id=self.rel2id)
        collect_logits = [[] for i in range(self.args.num_clients)]
        collect_tokens = []
        collect_mask = []
        for step, (labels, lengthes, tokens, mask) in enumerate(unlabel_val_loader):
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
            collect_tokens.append(tokens.cpu())
            collect_mask.append(mask.cpu())
            for i in range(len(self.clients)):
                with torch.no_grad():
                    self.clients[i].model.to(self.args.device)
                    self.clients[i].model.eval()
                    logits = self.clients[i].model(tokens, mask)
                    collect_logits[i].append(logits.cpu())
                    # self.clients[i].model.cpu()
        return collect_tokens, collect_mask, collect_logits

    def save_ckpt(self, save_path=None):
        if save_path is None:
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)
            for i in range(self.args.num_clients):
                torch.save(self.clients[i].model.state_dict(),
                           os.path.join(self.args.save_path, 'client_' + str(i) + '.pt'))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(self.args.num_clients):
                torch.save(self.clients[i].model.state_dict(),
                           os.path.join(save_path, 'client_' + str(i) + '.pt'))

    def load_ckpt(self, load_path=None):
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
                self.clients[i].model.load_state_dict(torch.load(load_path))

class Client(object):
    def __init__(self, args, model_name, train_data, test_data, rel2id):
        self.args = args
        self.model = get_model(model_name, rel2id).to(self.args.device)
        self.train_data = train_data
        self.test_data = test_data
        self.rel2id = rel2id
        self.model_name = model_name

    def KL_loss(self, inputs, target, reduction='average'):
        log_likelihood = F.log_softmax(inputs, dim=-1)
        target = F.softmax(target, dim=-1)

        if reduction == 'average':
            loss = F.kl_div(log_likelihood, target, reduction='mean')
        else:
            loss = F.kl_div(log_likelihood, target, reduction='sum')
        return loss

    def save_ckpt(self, id, save_path=None):
        if save_path is None:
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)
            torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'client_' + str(id) + '.pt'))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(), os.path.join(save_path, 'client_' + str(id) + '.pt'))

    def train(self):
        metric = 0.0
        if self.model_name == 'BertCLS':
            metric = self.BertCLS_train(self.model, self.train_data, self.test_data)
        elif self.model_name == 'BertMk':
            metric = self.BertCLS_train(self.model, self.train_data, self.test_data)
        elif self.model_name == 'BertCNN':
            metric = self.BertCNN_train(self.model, self.train_data, self.test_data)
        elif self.model_name == 'BertLSTM':
            metric = self.BertCNN_train(self.model, self.train_data, self.test_data)
        else:
            print("[Error] The model type does not exists !")
            assert 0
        return metric

    def distill_train(self, distill_loader):
        metric = 0.0
        if self.model_name == 'BertCLS':
            metric = self.BertCLS_distill_train(self.model, distill_loader, self.test_data)
        elif self.model_name == 'BertMk':
            metric = self.BertCLS_distill_train(self.model, distill_loader, self.test_data)
        elif self.model_name == 'BertCNN':
            metric = self.BertCNN_distill_train(self.model, distill_loader, self.test_data)
        elif self.model_name == 'BertLSTM':
            metric = self.BertCNN_distill_train(self.model, distill_loader, self.test_data)
        else:
            print("[Error] The model type does not exists !")
            assert 0
        return metric

    def smi_train(self, data_loader):
        metric = 0.0
        if self.model_name == 'BertCLS':
            metric = self.BertCLS_smi_train(self.model, data_loader, self.test_data)
        elif self.model_name == 'BertMk':
            metric = self.BertCLS_smi_train(self.model, data_loader, self.test_data)
        elif self.model_name == 'BertCNN':
            metric = self.BertCNN_smi_train(self.model, data_loader, self.test_data)
        elif self.model_name == 'BertLSTM':
            metric = self.BertCNN_smi_train(self.model, data_loader, self.test_data)
        else:
            print("[Error] The model type does not exists !")
            assert 0
        return metric


    def BertCLS_train(self, model, train_data, test_data):
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_data_loader = get_loader(self.args, train_data, self.rel2id)
        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': 0.00001},
                                {'params': model.classifier.parameters(), 'lr': 0.001}])
        for epoch in range(self.args.local_epoch):
            losses = []

            for step, (labels, lengthes, tokens, masks) in enumerate(train_data_loader):
                model.zero_grad()
                optimizer.zero_grad()
                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)
                logits = model(tokens, masks)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Finetuning loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric

    def BertCNN_train(self, model, train_data, test_data):
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_data_loader = get_loader(self.args, train_data, self.rel2id)
        no_decay = ['bias', 'LayerNorm', 'Layernorm.weight']
        param_optimizer = list(model.named_parameters())
        # print(param_optimizer)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=5e-5,
                             warmup=0.5,
                             t_total=len(train_data) * 10)
        for epoch in range(self.args.local_epoch):
            losses = []
            for step, (labels, lengthes, tokens, masks) in enumerate(train_data_loader):
                model.zero_grad()
                optimizer.zero_grad()
                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)
                logits = model(tokens, masks)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Finetuning loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric
    def BertCLS_smi_train(self, model, train_data, test_data):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': 0.00001},
                                {'params': model.classifier.parameters(), 'lr': 0.001}])
        for epoch in range(self.args.local_epoch):
            losses = []

            for step, (tokens, masks, labels) in enumerate(train_data):

                model.zero_grad()
                optimizer.zero_grad()
                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)
                logits = model(tokens, masks)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Finetuning loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric

    def BertCNN_smi_train(self, model, train_data, test_data):
        model.train()
        criterion = nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm', 'Layernorm.weight']
        param_optimizer = list(model.named_parameters())
        # print(param_optimizer)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=5e-5,
                             warmup=0.5,
                             t_total=300 * 10)
        for epoch in range(self.args.local_epoch):
            losses = []
            for step, (tokens, masks, labels) in enumerate(train_data):
                model.zero_grad()
                optimizer.zero_grad()
                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                masks = torch.stack([x.to(self.args.device) for x in masks], dim=0)
                logits = model(tokens, masks)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Finetuning loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric

    def BertCLS_distill_train(self, model, train_data, test_data):
        model.train()
        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': 0.00001},
                                {'params': model.classifier.parameters(), 'lr': 0.001}])
        for epoch in range(self.args.local_epoch):
            losses = []

            for step, (tokens, masks, target_logits) in enumerate(train_data):

                model.zero_grad()
                optimizer.zero_grad()
                tokens = tokens.to(self.args.device)
                masks = masks.to(self.args.device)
                target_logits = target_logits.to(self.args.device)
                logits = model(tokens, masks)

                loss = self.KL_loss(logits, target_logits)

                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Distillation loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric

    def BertCNN_distill_train(self, model, train_data, test_data):
        model.train()

        no_decay = ['bias', 'LayerNorm', 'Layernorm.weight']
        param_optimizer = list(model.named_parameters())
        # print(param_optimizer)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=5e-5,
                             warmup=0.5,
                             t_total=300 * 10)
        for epoch in range(self.args.local_epoch):
            losses = []
            for step, (tokens, masks, target_logits) in enumerate(train_data):
                model.zero_grad()
                optimizer.zero_grad()
                tokens = tokens.to(self.args.device)
                masks = masks.to(self.args.device)
                target_logits = target_logits.to(self.args.device)
                logits = model(tokens, masks)

                loss = self.KL_loss(logits, target_logits)


                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), self.args.max_grad_norm)
                optimizer.step()
            print(f'Distillation loss is {np.array(losses).mean()}')
        metric = self.Bert_evaluate(model, test_data)
        return metric

    def Bert_evaluate(self, model, test_data):
        model.eval()
        grnd = []
        pred = []
        metric = 0
        test_data_loader = get_loader(self.args, test_data, self.rel2id, batch_size=1)
        for step, (labels, lengthes, tokens, mask) in enumerate(test_data_loader):
            with torch.no_grad():
                labels = labels.to(self.args.device)
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
                logits = model(tokens, mask)
                max_id = np.argmax(logits.cpu().data.numpy(), axis=1).tolist()
                labels = labels.view(-1).detach().cpu().tolist()
                grnd += labels
                pred += max_id

        if self.args.metric == 'Acc':
            metric = get_acc(grnd, pred)
        elif self.args.metric == 'F1':
            metric = get_f1(grnd, pred, 0)
        return metric
        # model.eval()
        # n = len(test_data)
        # correct = 0
        # test_data_loader = get_loader(self.args, test_data, self.rel2id, batch_size=1)
        # for step, (labels, lengthes, tokens, mask) in enumerate(test_data_loader):
        #     labels = labels.to(self.args.device)
        #     tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
        #     mask = torch.stack([x.to(self.args.device) for x in mask], dim=0)
        #     logits = model(tokens, mask)
        #
        #     seen_sim = logits.cpu().data.numpy()
        #     max_smi = np.max(seen_sim, axis=1)
        #     label_sim = logits[:, labels].cpu().data.numpy()

        #     if label_sim >= max_smi:
        #         correct += 1
        # return correct / n
class RC(object):
    def __init__(self, args):
        self.args = args
        self.writer = args.writer
        if args.personal:
            self.train_data_list, self.labeled_valid_data, self.unlabeled_valid_data, self.test_data_lits = \
                get_data(args)
        else:
            self.train_data_list, self.labeled_valid_data, self.unlabeled_valid_data, self.test_data_lits = \
                get_data(args)
        self.rel2id, self.id2rel = get_relations(args)

        self.model_list = get_model_list(args)
        self.num_client = len(self.train_data_list)
        assert self.num_client == args.num_clients

    def add_dist_performace(self, i, metric, epoch):
        self.args.writer.add_scalar('Distill/Client' + str(i), metric, epoch)

    def add_local_performance(self, i, metric, epoch):
        self.args.writer.add_scalar('Local/Client' + str(i), metric, epoch)

    def add_loss(self, i, loss, epoch):
        self.args.writer.add_scalar('Loss/Client' + str(i), loss, epoch)

    def add_best_distill(self, mean, std, epoch):
        self.args.writer.add_scalar('Best_dist_mean', mean, epoch)
        self.args.writer.add_scalar('Best_dist_std', std, epoch)

    def add_best_local(self, mean, std, epoch):
        self.args.writer.add_scalar('Best_local_mean', mean, epoch)
        self.args.writer.add_scalar('Best_local_std', std, epoch)

