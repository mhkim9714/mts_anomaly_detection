import os
import argparse
import random
import pytz
from datetime import datetime
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from util.general import *
from util.data import *
import models
from config import *
from reconstruct import *
from forecast import *
from evaluate import *


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Multivariate Time Series Anomaly Detection')
    # data
    parser.add_argument('--train_path', type=str, default='./data/train.npy')
    parser.add_argument('--test_path', type=str, default='./data/test.npy')
    parser.add_argument('--label_path', type=str, default='./data/label.npy')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--n_sensor', type=int, default=30)
    # model
    parser.add_argument('--model', type=str, default='EncDec-AD') # EncDec-AD or Transformer
    # train
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    # log
    parser.add_argument('--log_dir', type=str, default='./results')
    parser.add_argument('--comment', type=str, default='comment')
    # misc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')

    args = parser.parse_args()
    fix_seed(args.seed)


    ##  Logging
    local_tz = pytz.timezone('Asia/Seoul')
    local_time = datetime.now(local_tz)

    if args.load_path == '':
        args.mode = 'train'
    else:
        args.mode = 'test'
    
    if args.mode == 'test':
        args.comment = 'test'
        log_name = local_time.strftime("%m-%d-%HH%MM%Ss")+f"{args.model}_{args.comment}"
        log_path = f'{args.load_path}/{log_name}/{log_name}.log'
        confusion_path = f'{args.load_path}/{log_name}'
        model_load_path = f'{args.load_path}/best_model.pt'
        if not os.path.exists(f'{args.load_path}/{log_name}'):
            os.mkdir(f'{args.load_path}/{log_name}')
    else:
        log_name = local_time.strftime("%m-%d-%HH%MM%Ss")+f"{args.model}_{args.comment}"
        log_path = f'{args.log_dir}/{log_name}/{log_name}.log'
        confusion_path = f'{args.log_dir}/{log_name}'
        model_save_path = f'{args.log_dir}/{log_name}/best_model.pt'
        if not os.path.exists(f'{args.log_dir}/{log_name}'):
            os.mkdir(f'{args.log_dir}/{log_name}')

    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s ::: %(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    logger = logging.getLogger('tqdm_logger')
    handler = TqdmLoggingHandler()
    logger.addHandler(handler)

    ### Load data -> sample data, already normalized #################################################################################
    train_data = np.load(args.train_path) # (10000,30)
    test_data = np.load(args.test_path) # (10000,30)
    label = np.load(args.label_path) # (10000,)

    val_data = train_data[-int(len(train_data)*args.val_ratio):]
    train_data = train_data[:-int(len(train_data)*args.val_ratio)]

    logger.info('*** DATA CONFIGURATION ***')
    logger.info(f"Train {train_data.shape} / Validation {val_data.shape} / Test {test_data.shape} / Test label {label.shape}")

    ## Apply time window
    final_train_data = generate_window(train_data, win_size = args.seq_len) # [total_window_cnt,seq_len,N]
    final_val_data = generate_window(val_data, win_size = args.seq_len)
    final_test_data = generate_window(test_data, win_size = args.seq_len)
    final_test_label = generate_window(label, win_size= args.seq_len) # [total_window_cnt,seq_len]

    train_dataset = TensorDataset(final_train_data, torch.zeros(final_train_data.shape[:-1])) # data / label
    val_dataset = TensorDataset(final_val_data, torch.zeros(final_val_data.shape[:-1]))
    test_dataset = TensorDataset(final_test_data, final_test_label)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ## Log arguments
    logger.info('*** ARGUMENT SETTINGS ***')
    for arg, value in vars(args).items():
        logger.info(f"Argument {arg} : {value}")

    ### Create model #########################################################################################################
    if args.model == 'EncDec-AD':
        cfg = model_configuration(args)
        model = models.LSTMAutoEncoder(input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'], seq_length=args.seq_len, num_layers=cfg['num_layers'])
    elif args.model == 'Transformer':
        cfg = model_configuration(args)
        model = models.Transformer(input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'], num_heads=cfg['num_heads'], t_past=args.seq_len, t_future=1)

    model.to(args.device)

    logger.info('*** MODEL CONFIGURATION ***')
    logger.info(model)

    ### Train to get the best model / Load the best model ####################################################################
    if cfg['type'] == 'reconstruct':
        if args.mode == 'train':
            train_reconstruct(args, cfg, model, train_loader, val_loader, logger, model_save_path)
            best_model = torch.load(model_save_path)
            best_model = best_model.to(args.device)
            logger.info(f"Best model loaded from {model_save_path}\n")
        elif args.mode == 'test':
            logger.info(f'Loading pretrained model...')
            best_model = torch.load(model_load_path)
            best_model = best_model.to(args.device)
            logger.info(f"Best model loaded from {model_load_path}\n")
        mean_error, std_error = normalize_anomaly_score_reconstruct(args, best_model, val_loader)
    
    elif cfg['type'] == 'forecast':
        if args.mode == 'train':
            train_transformer(args, cfg, model, train_loader, val_loader, logger, model_save_path)
            best_model = torch.load(model_save_path)
            best_model = best_model.to(args.device)
            logger.info(f"Best model loaded from {model_save_path}\n")
        elif args.mode == 'test':
            logger.info(f'Loading pretrained model...')
            best_model = torch.load(model_load_path)
            best_model = best_model.to(args.device)
            logger.info(f"Best model loaded from {model_load_path}\n")
        mean_error, std_error = normalize_anomaly_score_transformer(args, best_model, val_loader)
    
    ### Evaluate the anomaly detection performance #########################################################################
    AS_calculator = AnomalyScoring(mean_error, std_error)
    best_result = evaluation(args, cfg, best_model, test_loader, AS_calculator)

    ## 최종 best fb score를 갖는 AD 결과
    logger.info("*** Anomaly Detection Results ***")
    logger.info(f'Precision: {best_result[1]:.5f}')
    logger.info(f'Recall: {best_result[2]:.5f}')
    logger.info(f'F1-score: {best_result[3]:.5f}')
    logger.info(f'AUROC: {best_result[4]:.5f}')
    logger.info(f'Fb-score: {best_result[5]:.5f}')
    logger.info(f'Threshold: {best_result[0]:.3f}')



