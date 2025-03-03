from data_provider.data_loader import   Dataset_Custom,  Dataset_ETT_hour, Dataset_ETT_minute, Dataset_page_fault
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd
import torch, os

data_dict = {
    'custom': Dataset_Custom,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'page_fault': Dataset_page_fault,
}
def data_provider(args, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len
    
    shuffle_flag = False
    drop_last = drop_last_test
    batch_size = args.batch_size
    freq = args.freq
        
    data_set = Data(
        model_id=args.model_id , 
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    # get three copy of the same dataset
    train_data_set = data_set
    train_data_set.set_border('train')
    val_data_set = data_set
    val_data_set.set_border('val')
    test_data_set = data_set
    test_data_set.set_border('test')
    
    return train_data_set, val_data_set, test_data_set



def get_data_loader(train_data_set, val_data_set, test_data_set, args, shuffle_flag=True, drop_last=True):
    batch_size = args.batch_size
    
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return train_data_loader, val_data_loader, test_data_loader

