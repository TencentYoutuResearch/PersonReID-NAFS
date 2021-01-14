import os
import sys
import shutil
import time
import logging
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
from utils.metric import AverageMeter, Loss, constraints_loss
from test import test
from config import data_config, network_config, lr_scheduler, get_image_unique
from train_config import config
from tqdm import tqdm
import sys
from solver import WarmupMultiStepLR, RandomErasing

from test import test
import numpy as np
import re
import pickle
import random
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, 'best_model') + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, 'model_best', str(epoch)) + '.pth.tar'
        shutil.copyfile(filename, dst_best)


def train(epoch, train_loader, network, optimizer, compute_loss, args, co_location_loss=None):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    image_pre = AverageMeter()
    text_pre = AverageMeter()

    # switch to train mode
    network.train()

    end = time.time()

    for step, (images, captions, labels) in enumerate(train_loader):
        sep_captions = []
        
        n_sep = 2

        for i, c in enumerate(captions):
            c = re.split(r'[;,!?.]', c)
            if len(c) > n_sep or len(c) == n_sep:
                sep_captions = sep_captions + c[0:n_sep]
            else:
                pad_length = n_sep - len(c)
                padding = ["[PAD]" for j in range(pad_length)]
                sep_captions = sep_captions + c + padding

        tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions)
        sep_tokens, sep_segments, sep_input_masks, sep_caption_length = network.module.language_model.pre_process(sep_captions)

        tokens = tokens.cuda()
        segments = segments.cuda()
        input_masks = input_masks.cuda()
        caption_length = caption_length.cuda()

        sep_tokens = sep_tokens.cuda()
        sep_segments = sep_segments.cuda()
        sep_input_masks = sep_input_masks.cuda()
        
        images = images.cuda()
        labels = labels.cuda()

        p2 = [i for i in range(args.part2)]
        p3 = [i for i in range(args.part3)]
        random.shuffle(p2)
        random.shuffle(p3)

        global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = network(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3,  stage='train')

        cmpm_loss, cmpc_loss, cont_loss, loss, image_precision, text_precision, pos_avg_sim, neg_arg_sim, local_pos_avg_sim, local_neg_avg_sim = compute_loss(
            global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value, caption_length, labels)

        if step % 10 == 0:
            print('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}, cont_loss:{:.3f}, pos_sim_avg:{:.3f}, neg_sim_avg:{:.3f}, lpos_sim_avg:{:.3f}, lneg_sim_avg:{:.3f}'.
                  format(epoch, step, cmpm_loss, cmpc_loss, cont_loss, pos_avg_sim, neg_arg_sim, local_pos_avg_sim, local_neg_avg_sim))
        # constrain embedding with the same id at the end of one epoch
        if (args.constraints_images or args.constraints_text) and step == len(train_loader) - 1:
            con_images, con_text = constraints_loss(train_loader, network, args)
            loss += (con_images + con_text)

            print('epoch:{}, step:{}, con_images:{:.3f}, con_text:{:.3f}'.format(epoch, step, con_images.item(), con_text.item()))

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        train_loss.update(loss.item(), images.shape[0])
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, images.shape[0])
    return train_loss.avg, batch_time.avg, image_pre.avg, text_pre.avg


def main(args):

    # transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])



    cap_transform = None

    # data
    train_loader = data_config(args.image_dir, args.anno_dir, args.batch_size, 'train', 100, train_transform, cap_transform=cap_transform)

    test_loader = data_config(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform)  
    
    # loss
    compute_loss = Loss(args)
    nn.DataParallel(compute_loss).cuda()

    # network
    network, optimizer = network_config(args, 'train', compute_loss.parameters(), args.resume, args.model_path)

    # lr_scheduler
    scheduler = WarmupMultiStepLR(optimizer, (20, 25, 35), 0.1, 0.01, 10, 'linear')

    
    ac_t2i_top1_best = 0.0
    best_epoch = 0
    for epoch in range(args.num_epoches - args.start_epoch):
        network.train()
        # train for one epoch
        train_loss, train_time, image_precision, text_precision = train(args.start_epoch + epoch, train_loader, network, optimizer, compute_loss, args)

        # evaluate on validation set
        is_best = False
        print('Train done for epoch-{}'.format(args.start_epoch + epoch))

        logging.info('Epoch:  [{}|{}], train_time: {:.3f}, train_loss: {:.3f}'.format(args.start_epoch + epoch, args.num_epoches, train_time, train_loss))
        logging.info('image_precision: {:.3f}, text_precision: {:.3f}'.format(image_precision, text_precision))
        scheduler.step()
        for param in optimizer.param_groups:
            print('lr:{}'.format(param['lr']))

        if epoch >= 0:
            ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
        
            state = {'network': network.state_dict(), 'optimizer': optimizer.state_dict(), 'W': compute_loss.W, 'epoch': args.start_epoch + epoch}
           
            if ac_top1_t2i > ac_t2i_top1_best:
                best_epoch = epoch
                ac_t2i_top1_best = ac_top1_t2i
                save_checkpoint(state, epoch, args.checkpoint_dir, is_best)
            
            logging.info('epoch:{}'.format(epoch))
            logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))
       

    logging.info('Best epoch:{}'.format(best_epoch))
    logging.info('Train done')
    logging.info(args.checkpoint_dir)
    logging.info(args.log_dir)


if __name__ == "__main__":
    args = config()
    main(args)
