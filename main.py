import argparse
import torch
from exp.exp_parte import Exp_PARTE
import random
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
exp_dict = {'PARTE': Exp_PARTE}

#%%
parser = argparse.ArgumentParser(description='PARTE')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='Task type: pretrain for self-supervised representation learning, finetune for downstream forecasting.')
parser.add_argument('--is_training', type=int, default=1, help='Execution mode: 1 for training and 0 for evaluation or inference.')
parser.add_argument('--model_id', type=str, required=True, default='PARTE', help='Unique experiment identifier used for logging, checkpointing, and result saving.')
parser.add_argument('--model', type=str, required=True, default='PARTE', help='Model architecture name to instantiate.')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='Dataset name used for data loading and experiment configuration.')
parser.add_argument('--root_path', type=str, default='datasets/', help='Root directory containing all dataset files.')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Dataset file name located under the specified root directory.')
parser.add_argument('--features', type=str, default='M', help='Forecasting setting: M=multivariate-to-multivariate, S=univariate-to-univariate, MS=multivariate-to-univariate.')
parser.add_argument('--target', type=str, default='OT', help='Target variable name for S or MS forecasting settings.')
parser.add_argument('--freq', type=str, default='h', help='Sampling frequency used for temporal feature encoding, e.g., h, d, w, m, 15min, or 3h.')
parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/', help='Directory for saving fine-tuning checkpoints.')
parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/', help='Directory for saving self-supervised pre-training checkpoints.')
parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth', help='Pre-trained checkpoint file used to initialize fine-tuning.')
parser.add_argument('--load_checkpoints', type=str, default=None, help='Checkpoint path used for resuming training or performing evaluation.')
parser.add_argument('--scale', action='store_false', help='Scale dataset')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='Length of the historical input sequence used as model observations.')
parser.add_argument('--label_len', type=int, default=0, help='Length of decoder start tokens retained from the input sequence; unused in PARTE.') 
parser.add_argument('--pred_len', type=int, default=96, help='Length of the future forecasting horizon to predict.')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='Number of input variables fed into the encoder.')
parser.add_argument('--dec_in', type=int, default=7, help='Number of decoder input variables; retained for compatibility with Transformer-based implementations.')
parser.add_argument('--c_out', type=int, default=7, help='Number of output variables to forecast.')
parser.add_argument('--d_model', type=int, default=512, help='Dimension of latent feature representations.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads in multi-head attention modules.')
parser.add_argument('--e_layers', type=int, default=2, help='Number of stacked encoder layers.')
parser.add_argument('--d_ff', type=int, default=2048, help='Hidden dimension of the feed-forward network.')
parser.add_argument('--d_hidden', type=int, default=128, help='Hidden dimension used in frequency-domain feature learning modules.')
parser.add_argument('--factor', type=int, default=1, help='Attention sparsity factor used in efficient attention mechanisms.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate applied throughout the model.')
parser.add_argument('--fc_dropout', type=float, default=0, help='Dropout rate applied to fully connected layers.')
parser.add_argument('--head_dropout', type=float, default=0.1, help='Dropout rate applied to prediction heads.')
parser.add_argument('--embed', type=str, default='timeF', help='Temporal embedding method: timeF, fixed, or learned.')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function used in nonlinear layers.')
parser.add_argument('--output_attention', action='store_true', help='Return attention maps during forward propagation.')
parser.add_argument('--individual', type=int, default=0, help='Use variable-specific prediction heads: 1=True, 0=False.')
parser.add_argument('--pct_start', type=float, default=0.3, help='Percentage of training steps used for learning-rate warmup in OneCycleLR.')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes used for data loading.')
parser.add_argument('--itr', type=int, default=1, help='Number of repeated experiment runs with different random seeds.')
parser.add_argument('--train_epochs', type=int, default=10, help='Maximum number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size used during training.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate for the optimizer.')
parser.add_argument('--patience', type=int, default=3, help='Number of epochs to wait before early stopping when validation performance does not improve.')
parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy.')
parser.add_argument('--use_amp', action='store_true', default=False, help='Enable automatic mixed precision training to reduce memory usage and accelerate computation.')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU acceleration for training and inference.')
parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU device to use.')
parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Enable distributed training across multiple GPUs.')
parser.add_argument('--devices', type=str, default='0', help='Comma-separated GPU device IDs, e.g., "0", "0,1", or "0,1,2,3".') 

# Pre-train
parser.add_argument('--kernel_size', type=int, default=25, help='Moving-average kernel size used for trend extraction during time-series decomposition.')
parser.add_argument('--seg_len', type=int, default=24, help='Length of each contiguous masking segment in self-supervised pre-training.')
parser.add_argument('--p_tmask', type=float, default=0.2, help='Masking ratio applied to the trend component during pre-training.')
parser.add_argument('--topk', type=int, default=2)

# TDA layer setting and contrastive learning
parser.add_argument('--use_tda', action='store_true', help='Enable topology-aware representation learning based on persistent homology.')
parser.add_argument('--n_kernels', type=int, default=8, help='Number of learnable topological kernels used in the TPB layer.')
parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter used in contrastive learning objectives.')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight assigned to the contrastive loss in the overall training objective.')
parser.add_argument('--tda_pooling', type=str, default='sum', help='Pooling strategy used to aggregate responses from multiple topological kernels.')

# reproduce
parser.add_argument('--seed', default=1000, type=int)


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print('torch.cuda.device_count()', torch.cuda.device_count())
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
print('Args in experiment:')
print(args)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

Exp = exp_dict[args.model]
if args.task_name == 'pretrain' and args.model == 'PARTE':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_dh{}_el{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ks{}_sl{}_pt{}_topk{}_tau{}_alp{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.d_hidden,
            args.e_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.kernel_size,
            args.seg_len,
            args.p_tmask,
            args.topk,
            args.tau,
            args.alpha,
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.pretrain()
        torch.cuda.empty_cache()
        
elif args.task_name == 'finetune':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_dh{}_el{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ks{}_sl{}_pt{}_topk{}_tau{}_alp{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.d_hidden,
            args.e_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.kernel_size,
            args.seg_len,
            args.p_tmask,
            args.topk,
            args.tau,
            args.alpha,
        )
        
        args.load_checkpoints = os.path.join(args.pretrain_checkpoints, args.data, args.transfer_checkpoints)
        exp = Exp(args)  # set experiments
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

# %%

