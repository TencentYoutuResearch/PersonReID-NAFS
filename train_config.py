import argparse
import os
import logging
from config import log_config, dir_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')

    # Directory
    parser.add_argument('--image_dir', type=str, help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, help='directory to store anno file')
    parser.add_argument('--checkpoint_dir', type=str, help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str, help='directory to store log')
    parser.add_argument('--model_path', type=str, default = None, help='directory to pretrained model, whole model or just visual part')


    # Model setting
    parser.add_argument('--resume', action='store_true', help='whether or not to restore the pretrained whole model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--ckpt_steps', type=int, default=5000, help='#steps to save checkpoint')
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--CMPM', action='store_true')
    parser.add_argument('--CMPC', action='store_true')
    parser.add_argument('--CONT', action='store_true')
    parser.add_argument('--focal_type', type=str, default=None)
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)
    parser.add_argument('--constraints_text', action='store_true')
    parser.add_argument('--constraints_images', action='store_true')
    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--pretrained', action='store_true', help='whether or not to restore the pretrained visual model')
    parser.add_argument('--part2', type=int, default=3)
    parser.add_argument('--part3', type=int, default=2)
    parser.add_argument('--lambda_softmax', type=float, default=20.0, help='scale constant')
    parser.add_argument('--lambda_cont', type=float, default=0.2, help='hyper-parameter of contrastive loss')
    parser.add_argument('--reranking', action='store_true', help='whether reranking during testing')
    
    # Optimization setting
    parser.add_argument('--optimizer', type=str, default='adam', help='one of "sgd", "adam", "rmsprop", "adadelta", or "adagrad"')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--wd', type=float, default=0.00004)
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--end_lr', type=float, default=0.0001, help='minimum end learning rate used by a polynomial decay learning rate')
    parser.add_argument('--lr_decay_type', type=str, default='exponential', help='One of "fixed" or "exponential"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='50,100', help='#epoches when learning rate decays')

    parser.add_argument('--nsave', type=str, default='')


    # Default setting
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()
    return args


def config():
    args = parse_args()
    dir_config(args)
    log_config(args,'train')
    return args
