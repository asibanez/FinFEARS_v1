# Utilities
import os
import json
import torch
import logging
import datetime
import argparse
import torch.nn as nn

def parse_args_f():
    parser = argparse.ArgumentParser()
    # Task arguments
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                        help = 'input folder')
    parser.add_argument('--output_dir', default = None, type = str, required = True,
                        help = 'output folder')
    parser.add_argument('--task', default = None, type = str, required = True,
                        help = 'number of total epochs to run')
    
    # Common arguments
    parser.add_argument('--model_name', default = None, type = str, required = True,
                        help = 'BERT model name')
    parser.add_argument('--seq_len', default = None, type = int, required = True,
                        help = 'text sequence length')
    parser.add_argument('--hidden_dim', default = None, type = int, required = True,
                        help = 'lstm hidden dimension')
    parser.add_argument('--alpha', default = None, type = float, required = True,
                        help = 'alpha coefficient')
    parser.add_argument('--att_dim', default = None, type = int, required = True,
                        help = 'self-attention dimension')
    parser.add_argument('--freeze_BERT', default = None, type = str, required = True,
                        help = 'freeze BERT layer')
    parser.add_argument('--seed', default = None, type = int, required = True,
                        help = 'random seed')
    parser.add_argument('--use_cuda', default = None, type = str, required = True,
                        help = 'use CUDA')
    
    # Train arguments
    parser.add_argument('--n_epochs', default = None, type = int, required = True,
                        help = 'number of total epochs to run')
    parser.add_argument('--batch_size_train', default = None, type = int, required = True,
                        help = 'train batch size')
    parser.add_argument('--shuffle_train', default = None, type = str, required = True,
                        help = 'shuffle train set')   
    parser.add_argument('--drop_last_train', default = None, type = str, required = True,
                        help = 'Drop last batch from train set')    
    parser.add_argument('--dev_train_ratio', default = None, type = int, required = True,
                        help = 'size dev set / size train set')    
    parser.add_argument('--train_toy_data', default = None, type = str, required = True,
                        help = 'Use toy dataset for training')
    parser.add_argument('--len_train_toy_data', default = None, type = int, required = True,
                        help = 'train toy data size')
    parser.add_argument('--lr', default = None, type = float, required = True,
                        help = 'learning rate')
    parser.add_argument('--wd', default = None, type = float, required = True,
                        help = 'weight decay')
    parser.add_argument('--dropout', default = None, type = float, required = True,
                        help = 'dropout')
    parser.add_argument('--momentum', default = None, type = float, required = True,
                        help = 'momentum')
    parser.add_argument('--save_final_model', default = None, type = str, required = True,
                        help = 'final .pt model is saved in output folder')
    parser.add_argument('--save_model_steps', default = None, type = str, required = True,
                        help = 'intermediate .pt models saved in output folder')
    parser.add_argument('--save_step_cliff', default = None, type = int, required = True,
                        help = 'start saving models after cliff')
    parser.add_argument('--gpu_ids_train', default = None, type = str, required = True,
                        help='gpu IDs for training')
    
    # test argments
    parser.add_argument('--test_file', default = None, type = str, required = True,
                        help='test datset filename')
    parser.add_argument('--model_file', default = None, type = str, required = True,
                        help='trained model filename')
    parser.add_argument('--batch_size_test', default = None, type = str, required = True,
                        help='test batch size')
    parser.add_argument('--gpu_id_test', default = None, type = str, required = True,
                        help='gpu ID for testing')
    
    args = parser.parse_args()

    return args

def make_dir_f(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created folder : ", path)

def get_logger_f(output_dir):
    path_log_file = os.path.join(output_dir, 'log.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(filename = path_log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def save_args_f(args):
    output_path_params = os.path.join(args.output_dir, 'params.json')
    with open(output_path_params, 'w') as fw:
        json.dump(vars(args), fw)

def model_2_device_f(args, model):
    if eval(args.use_cuda) and torch.cuda.is_available():
        if args.task == 'Train':
            gpu_ids = [int(x) for x in args.gpu_ids_train.split(',')]
        if args.task == 'Test':
            gpu_ids = [int(args.gpu_id_test)]
        print('Moving model to cuda')
        # DataParallel training if multiple GPU and train task
        if len(gpu_ids) > 1 and args.task == 'Train':
            device = torch.device('cuda', gpu_ids[0])
            model = nn.DataParallel(model, device_ids = gpu_ids)
            model = model.cuda(device)
        else:
            device = torch.device('cuda', gpu_ids[0])
            model = model.cuda(device)
        print('Done')
    else:
        device = torch.device('cpu')
        model = model.to(device)

    return model, device

def save_checkpoint_f(args, epoch, model, optimizer, train_loss):
    output_path_model = os.path.join(args.output_dir, 'model.pt')
    if len(args.gpu_ids_train) > 1 and eval(args.use_cuda) and torch.cuda.is_available():
        model_state_dict_save = model.module.state_dict()
    else:
        model_state_dict_save = model.state_dict()
    torch.save({'epoch': epoch,
                'model_state_dict': model_state_dict_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss},
                output_path_model + '.' + str(epoch))
    
def save_train_results_f(args, train_loss_history, val_loss_history, start_time):
    output_path_results = os.path.join(args.output_dir, 'train_results.json')
    end_time = datetime.datetime.now()
    results = {'training_loss': train_loss_history,
               'validation_loss': val_loss_history,
               'start time': str(start_time),
               'end time': str(end_time)}
    with open(output_path_results, 'w') as fw:
        json.dump(results, fw)

def save_test_results_f(args, avg_loss):
    output_path_results = os.path.join(args.output_dir, 'test_results.json')
    results = {'avg_loss': avg_loss}
    with open(output_path_results, 'w') as fw:
        json.dump(results, fw)
