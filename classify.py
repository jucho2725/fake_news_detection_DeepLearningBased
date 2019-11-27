'''
for test new data
input: model, data, weight_matrix
output: accuracy, loss, plot
'''
import argparse

import numpy as np
import torch

from model.bi_lstm import BiLSTM
from trainer.BaseDL import Trainer


def define_argparser():
    # argparse
    parser = argparse.ArgumentParser(description = 'run argparser')
    parser.add_argument('--model',required=False,default='bi-lstm', help='select model')
    parser.add_argument('--weights_matrix',required=False,default = 'data/weights_matrix_6B_300.npy', help='weights matrix path for word embeddings')
    parser.add_argument('--model_path',required=False, default = 'data/best_model.pt', help='model checkpoint path')

    parser.add_argument('--sent_pad_path',required=False, default ='data/sent_pad_modified.npy', help='padded sentence(preprocessed)')
    parser.add_argument('--label_path',default='data/label_modified.pkl')
    parser.add_argument('--data_path',required=False, default = '', help='fake news data path (csv format), must include text, type columns')

    args = parser.parse_args()
    return args

def main(args):
    # gpu 하나일 때 / colab 기준 환경
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(2019)
    if device == 'cuda':
        torch.cuda.manual_seed_all(2019)


    if args.model =='bi-lstm':
        weights_matrix = np.load(args.weights_matrix, allow_pickle=True)  # 새로 저장
        model = BiLSTM(weights_matrix).to(device)


    elif args.model =='cnn':
        pass

    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict=state_dict)

    cls = Trainer(args)
    cls.test(model, device)

    """ EAN """
    # criterion = nn.BCELoss().to(device)
    # test(model, te_dataloader, criterion, device)

if __name__ =='__main__':
    args =define_argparser()
    main(args)

