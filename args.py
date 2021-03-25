import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--exp_dir', type=str, default='')
parser.add_argument('--dataset_file', type=str, default='./raw_data.xlsx')
parser.add_argument('--sheet_name', type=str, default='S&P500 Index Data', choices=['S&P500 Index Data',])
parser.add_argument('--src_seq_len', type=int, default=30)
parser.add_argument('--tgt_seq_len', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=10)

parser.add_argument('--train_proportion', type=float, default=0.8)
parser.add_argument('--valid_proportion', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--grad_bound', type=float, default=10.0)

parser.add_argument('--lr', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0.00001)
parser.add_argument('--l2_reg', type=float, default=1e-4)

parser.add_argument('--encoder_nlayers', type=int, default=6)
parser.add_argument('--encoder_nhead', type=int, default=8)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--nhid', type=int, default=2048)
parser.add_argument('--decoder_nlayers', type=int, default=4)
parser.add_argument('--decoder_nhead', type=int, default=8)
parser.add_argument('--dropout', type=int, default=0.5)

parser.add_argument('--disable_cuda', default=False, action='store_true')

args = parser.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else: 
    args.device = torch.device('cpu')

