from REModels.BertCLS_model import Bert_CLS_model
from REModels.BertMk_model import Bert_Mark_model
from REModels.BertCNN_model import Bert_CNN_model
from REModels.BertLSTM_model import Bert_LSTM_model
from REModels.BertEntCLS_model import Bert_EntCLS_model
from REModels.BertEntLSTM_model import Bert_EntLSTM_model
from REModels.BertEntMk_model import BertEnt_model
from argparse import ArgumentParser
from copy import deepcopy

class BertCLS_config(object):
    def __init__(self):
        # data
        self.num_of_relation = 40
        # bert
        self.bert_path = 'bert-base-uncased'
        self.encoder_output_size = 768
        self.max_length = 256
        self.vocab_size = 30522
        self.marker_size = 4
        self.pattern = 'entity_marker'
        # training
        self.drop_out = 0.5

class BertMk_config(object):
    def __init__(self):
        # data
        self.num_of_relation = 40
        # bert
        self.bert_path = 'bert-base-uncased'
        self.encoder_output_size = 768
        self.max_length = 256
        self.vocab_size = 30522
        self.marker_size = 4
        self.pattern = 'entity_marker'
        # training
        self.drop_out = 0.5

class BertCNN_config(object):
    def __init__(self):
        # data
        self.num_of_relation = 40
        # bert
        self.bert_path = 'bert-base-uncased'
        self.encoder_output_size = 768
        self.max_length = 256
        self.vocab_size = 30522
        self.marker_size = 4
        self.pattern = 'entity_marker'
        # cnn
        self.cnn_hidden_size = 768
        # training
        self.drop_out = 0.5

class BertLSTM_config(object):
    def __init__(self):
        # data
        self.num_of_relation = 40
        # bert
        self.bert_path = 'bert-base-uncased'
        self.encoder_output_size = 768
        self.max_length = 256
        self.vocab_size = 30522
        self.marker_size = 4
        self.pattern = 'entity_marker'
        # cnn
        self.lstm_hidden_size = 768
        # training
        self.drop_out = 0.2

def get_BertCLS_model(config):
    CLS_model = Bert_CLS_model(config)
    return CLS_model

def get_BertMk_model(config):
    Mk_model = Bert_Mark_model(config)
    return Mk_model

def get_BertCNN_model(config):
    CNN_model = Bert_CNN_model(config)
    return CNN_model

def get_BertEntLSTM_model(config):
    LSTM_model = Bert_EntLSTM_model(config)
    return LSTM_model
def get_BertEntCLS_model(config):
    CLS_model = Bert_EntCLS_model(config)
    return CLS_model

def get_BertEntMk_model(config):
    Mk_model = BertEnt_model(config)
    return Mk_model

def get_BertEntCNN_model(config):
    CNN_model = Bert_EntLSTM_model(config)
    return CNN_model

def get_BertLSTM_model(config):
    LSTM_model = Bert_LSTM_model(config)
    return LSTM_model

def get_model_list(args):
    model_type = args.model_types
    num_clients = args.num_clients
    if args.personal:
        model_name_list = [model_type[i % len(model_type)] for i in range(num_clients)]
    else:
        model_name_list = [model_type[1] for i in range(num_clients)]
    return model_name_list

def get_model(name, rel2id):

    if name == 'BertCLS':
        config = BertCLS_config()
        config.num_of_relation = len(rel2id)
        return get_BertCLS_model(config)

    elif name == 'BertMk':
        config = BertMk_config()
        config.num_of_relation = len(rel2id)
        return get_BertMk_model(config)

    elif name == 'BertCNN':
        config = BertCNN_config()
        config.num_of_relation = len(rel2id)
        return get_BertCNN_model(config)

    elif name == 'BertLSTM':
        config = BertLSTM_config()
        config.num_of_relation = len(rel2id)
        return get_BertLSTM_model(config)

    else:
        print("[Error] The model type does not exists !")
        assert 0

def get_ente_model(name, rel2id, ent2id):
    if name == 'BertCLS':
        config = BertCLS_config()
        config.num_of_relation = len(rel2id)
        config.num_of_entity = len(ent2id)
        return get_BertEntCLS_model(config)

    elif name == 'BertMk':
        config = BertMk_config()
        config.num_of_relation = len(rel2id)
        config.num_of_entity = len(ent2id)
        return get_BertEntMk_model(config)

    elif name == 'BertCNN':
        config = BertCNN_config()
        config.num_of_relation = len(rel2id)
        config.num_of_entity = len(ent2id)
        return get_BertEntCNN_model(config)

    elif name == 'BertLSTM':
        config = BertLSTM_config()
        config.num_of_relation = len(rel2id)
        config.num_of_entity = len(ent2id)
        return get_BertEntLSTM_model(config)

    else:
        print("[Error] The model type does not exists !")
        assert 0