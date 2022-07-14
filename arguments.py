
def add_arguments(parser):

    # federated data
    parser.add_argument('--dataset_name', default='tacred', type=str, help='dataset name: ')
    parser.add_argument('--niid_type', default='quantity', type=str, help='the niid type: quantity or label')
    parser.add_argument('--niid_alpha', default=1, type=int, help='the hp of niid distribution')
    parser.add_argument('--valid_shot', default=15, type=int, help='the number of labeled valid data')
    parser.add_argument('--max_length', default=256, type=int, help='the max length of sentence')
    parser.add_argument('--early_stop', default=25, type=int, help='the bad count for optimization')

    # training
    parser.add_argument('--device', default='cuda', type=str, help='the training device')
    parser.add_argument('--max_grad_norm', default=10, type=int, help='clip the grad norm')
    parser.add_argument('--num_workers', default=2, type=int, help='')

    # federated settings
    parser.add_argument('--local_epoch', default=1, type=int, help='local training epochs')
    parser.add_argument('--global_epoch', default=40, type=int, help='global training epochs')
    parser.add_argument('--fraction', default=1.0, type=float, help='the ratio of selected clients')
    parser.add_argument('--batch_size', default = 16, type=int, help='the batch size of training')

    parser.add_argument('--personal', default=True, type=bool, help='whether the test mode is personlized')

    # RC models
    parser.add_argument('--model_types', default=['BertCLS', 'BertMk', 'BertLSTM'], type=list, help='RC models')
    parser.add_argument('--model_heter', default=True, type=bool, help='whether the models are heterogeneus')
    parser.add_argument('--metric', default='Acc', type=str, help='Choose from: Acc and F1')
    parser.add_argument('--load_ckpt', default=False, type=bool, help='whether to load ckpt')

    # for memory
    parser.add_argument('--input_size', default=768, type=int, help='the size of proto')
    parser.add_argument('--num_relations', default=41, type=int, help='the number of relations')

