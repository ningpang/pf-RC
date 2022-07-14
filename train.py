from argparse import ArgumentParser
import json
import os
from arguments import add_arguments
from Frameworks.PFRC import PFRC
from torch.utils.tensorboard import SummaryWriter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = ArgumentParser(description='Personalized Federated Relation Classification')
    add_arguments(parser)
    parser.add_argument('--num_clients', default=10, type=int, help='the number of clients')
    parser.add_argument('--fed_mode', default='pfrc', type=str, help='choose from: pfrc')
    args = parser.parse_args()
    if args.personal:
        per = 'pFed'
    else:
        per = 'nonpFed'

    args.pretrain_path = 'pretrain_' + per + '_' + args.fed_mode + '_' + args.niid_type + '_' + str(
        args.num_clients) + '_' + str(args.niid_alpha)
    args.save_path = 'save_ckpt_' + per + '_' + args.fed_mode + '_' + args.niid_type + '_' + str(
        args.num_clients) + '_' + str(args.niid_alpha)
    args.pretrain_path = os.path.join(args.dataset_name, args.pretrain_path)
    args.save_path = os.path.join(args.dataset_name, args.save_path)

    args_str = json.dumps(vars(args))
    print(args_str)
    fed_mode = '-'.join([str(args.fed_mode), str(args.niid_type), str(args.num_clients), str(args.niid_alpha)])
    print(f'Federated Learning: {fed_mode}')
    writer = SummaryWriter(os.path.join('Summary/' + args.dataset_name, fed_mode))
    args.writer = writer

    if args.fed_mode == 'pfrc':
        FRC_learner = PFRC(args)
        FRC_learner.train()
    else:
        print("[Error] The federated mode does not exists !")
        assert 0

