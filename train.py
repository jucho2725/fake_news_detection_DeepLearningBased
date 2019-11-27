import argparse

import numpy as np
import torch

import json

from model.bi_lstm import BiLSTM
from model.ean import init_model
from trainer.BaseDL import Trainer as Trainer_base
from trainer.EAN import Trainer as Trainer_ean


def load_config():
    with open('cfg/bi_lstm.json', 'r') as f:
        b_cfg = json.load(f)
    with open('cfg/ean.json', 'r') as f:
        e_cfg = json.load(f)
    return b_cfg, e_cfg

def define_argparser():
    '''
    Define argument parser
    :return: configuration object
    '''

    # NOTE : We assume that the dataset is not separated.
    p = argparse.ArgumentParser(description = 'check model')
    p.add_argument('--model',required=False,default='bi-lstm', help='select model => either bi-lstm, cnn, EAN')
    p.add_argument('--save_dir', required=False, help='where to save model checkpoint')

    """ params """
    p.add_argument('--batch_size', type=int, default= 64)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--hidden_size', type=int, default=100)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--n_class', type=int, default=1, help="We only implement binary classification case. Multiclass will be updated soon.")
    p.add_argument('--learning rate', type=float, required=False, default=1e-5)

    p.add_argument('--val_every', type=int, default=1)
    args = p.parse_args()
    return args

def main(args, cfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(2019)
    if device == 'cuda':
        torch.cuda.manual_seed_all(2019)

    if args.model == 'bi-lstm':
        weights_matrix = np.load(cfg['weights_matrix'], allow_pickle=True)  # 새로 저장
        model = BiLSTM(weights_matrix).to(device)
        trainer = Trainer_base(cfg)
        trainer.train(num_epochs=args.n_epochs,
                  model=model,
                  saved_dir=args.save_dir,
                  device=device,
                  criterion=torch.nn.BCELoss(),
                  optimizer=torch.optim.Adam(params=model.parameters(), lr = 1e-5),
                  val_every=args.val_every
                  )

    # TO DO
    elif args.model == 'cnn':
        pass

    elif args.model == 'EAN':
        model = init_model(N=9, d_model=768, d_ff=3072, head=12, dropout=0.1).to(device)
        trainer = Trainer_ean(cfg)
        trainer.train(num_epochs=args.n_epochs,
                  model=model,
                  saved_dir=args.save_dir,
                  device=device,
                  criterion=torch.nn.BCELoss(),
                  optimizer=torch.optim.Adam(params=model.parameters(), lr = 1e-5),
                  val_every=args.val_every
                  )



if __name__ == '__main__':
    args = define_argparser()

    b_cfg, e_cfg = load_config()
    if args.model == "bi-lstm":
        main(args, b_cfg)
    elif args.model == "cnn":
        pass
    elif args.model == "ean": # EAN
        main(args, e_cfg)

