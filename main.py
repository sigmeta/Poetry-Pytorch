import torch
import torch.nn as nn
from models.model import PTransformer,TransformerLM,LSTMLM
from models.trainer import Trainer_LSTM, Trainer_Transformer, Trainer_TransformerLM
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
from data.process import TextConverter,get_dataset
import argparse
import os

def Opt():
    parser = argparse.ArgumentParser(description="Argparse of Poetry-Pytorch")
    #path
    parser.add_argument('-p','--data_path', default='dataset/poetry-no.txt')
    parser.add_argument('-o','--output_path', default='output/model.pkl')
    #model config
    parser.add_argument('--model', default='lstm', choices=['transformer','transformerlm','lstm'])
    parser.add_argument('--hidden_dims', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=1, help="Transformer encoder layers")
    parser.add_argument('--num_decoder_layers', type=int, default=2, help="Transformer decoder layers")
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # training params
    parser.add_argument('--train', action='store_true',default=False)
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--src_max_len', type=int, default=8)
    parser.add_argument('--tgt_max_len', type=int, default=16)
    parser.add_argument('--src_text', default="春", help="used when --model=transformer, it is like a title")
    parser.add_argument('--tgt_text', default="", help="a start of the poetry")
    args = parser.parse_args()
    return args

def main():
    opt=Opt()
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    convert = TextConverter(opt.data_path)
    if opt.model=='lstm':
        trainer=Trainer_LSTM(convert,opt)
    elif opt.model=='transformer':
        trainer=Trainer_Transformer(convert,opt)
    elif opt.model=='transformerlm':
        trainer=Trainer_TransformerLM(convert,opt)

    if opt.train:
        trainer.train()
    if opt.test:
        trainer.test()


if __name__=="__main__":
    main()
