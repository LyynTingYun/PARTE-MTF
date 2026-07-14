from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from tda_module.tda_collate import TDACollateFn
import torch
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'weather': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Exchange' : Dataset_Custom,
    'ILI': Dataset_Custom,
    'c46207':Dataset_Custom,
    'c46131':Dataset_Custom,
    'c46132':Dataset_Custom,
    'c46036':Dataset_Custom,
    'c46204':Dataset_Custom,
    'c46205':Dataset_Custom,
    'c46206':Dataset_Custom,
    'c46134':Dataset_Custom,
    'c46146':Dataset_Custom,
    'c46145':Dataset_Custom,
}
    
def data_provider(args, flag):
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True if flag == 'train' else False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn = TDACollateFn() if args.use_tda else None
    )
    
    print(flag, len(data_set), len(data_loader))
    return data_set, data_loader


