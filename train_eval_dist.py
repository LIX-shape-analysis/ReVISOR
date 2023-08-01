"""
This script trains and evaluate a model using EDF data or an external data source.
"""
import argparse
import glob
import os
import sys
import time
import json
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import data_utils as d_utils
from models import build_scene_segmentation
from outlier_segmentation_dataset import OutlierSegmentationDataset
from utils.util import AverageMeter, get_metrics_and_print, get_metrics_train_and_print
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from utils.config import config, update_config

from data_utils import read_ply_ls, write_ply, create_dir_if_required

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from utils.util import get_metrics_dict, print_metric_dict

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import pickle
import pandas as pd

# Metrics monitored during training:
# prec: precision, rec: recall, f_b: f_beta score, miou: mean intersection over union.
SAVED_METRICS = ["prec","rec","f_b","miou"]

def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / e_x.sum(axis=axis,keepdims=True)


def parse_option():
    print("NUM AVAILABLE GPUS: {}".format(torch.cuda.device_count()))
    parser = argparse.ArgumentParser('Outlier segmentation training')

    parser.add_argument('--intensity', type=int, default=0, help='using intensity (1) or not (0). Default: 0')
    parser.add_argument('--visibility', type=int, default=0, help='using visibility (1) or not (0). Default: 0')
    parser.add_argument('--katz_type', type=str, default='', help='Type of visibility inversion function')
    parser.add_argument('--katz_params', action='append', default=[], help='List of Katz parameters')


    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')

    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--lr_decay_steps', type=int, help='lr decay steps')
    parser.add_argument('--lr_decay_rate', type=float, help='lr decay rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')

    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')
    parser.add_argument('--job_name',type=str,required=True)
    parser.add_argument('--original_warmup_name',type=str,required=True)

    parser.add_argument('--DEBUG', type=int, required=True, help='Whether to debug or not (i.e. 1 ply file in datasets)')
    parser.add_argument('--from_scratch',type=int,required=True, help='If we resume a training or not.')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset type ("PCN" or "EDFS" or "EDFL")')
    parser.add_argument('--local_aggregator', type=str, required=True, help='Local aggregator type ("pseudogrid", "pospool", "adaptiveweight", "pointwisemlp", "minkowski")')
    parser.add_argument('--diameter_percent', type=int, required=True, help='Dataset type ("PCN" or "EDFS" or "EDFL")')

    # data augmnetations
    parser.add_argument('-ar','--aug_rot', action='append', help='List of rotation augmentation angles (in rad)', required=True)
    parser.add_argument('-as','--aug_sym', action='append', help='List of symetry augmentation axis (bool)', required=True)

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()


    local_aggregator = args.local_aggregator

    # infer cfg from local aggregator given ("pool","grid","adap","mlp")
    cfg = glob.glob("./cfgs/*{}*.yaml".format(local_aggregator))[0]

    update_config(cfg)

    config.features = []
    config.katz_params = []
    if args.intensity==1:
        config.features.append("intensity")
    if args.visibility==1:
        config.features.append("visibility")
        config.katz_params = [float(kp) for kp in args.katz_params]
        config.katz_type = args.katz_type

    config.local_aggregator = local_aggregator

    config.dataset = args.dataset

    if "EDF" in config.dataset:
        shape_diameter = 10. # 5m radius spherical scenes
        config.sampleDl = 0.

        # Path to be changed to retrieve the point clouds to be processed.
        config.data_root = "/data/train/"


    else:
        shape_diameter = 1. # normalized shapes
        config.sampleDl = config.in_radius/32. # in total, 32 upsamples --> first layer = pointwise mlp (no downsampling)
        config.data_root = "/path/to/data/"



    config.DEBUG = args.DEBUG

    # receptive field size
    config.in_radius = 0.5*shape_diameter*args.diameter_percent/100.

    # subsamplings
    if config.in_radius>=0.25:
        config.sampleDl = 0.02
    if config.in_radius>=1:
        config.sampleDl = 0.04
    if config.in_radius>1.5:
        config.sampleDl = 0.06
    if config.in_radius>2.1:
        config.sampleDl = 0.09

    config.radius = max(config.in_radius*np.sqrt(3)/32.,0.025)

    if args.num_points==15000: # cross-validation dans CloserLook3D
        config.nsamples = [26,31,38,41,39]
        config.npoints = [4096,1152,304,88]
    else: # si on utilise moins de 15000, setup recommandée :
        config.nsamples = [2*26,int(1.5*26),int(1.25*26),26,26]
        config.npoints = [max(int(args.num_points/4.),1),max(int(args.num_points/16.),1),max(int(args.num_points/32.),1),max(int(args.num_points/128.),1)]



    print("RADII: in_radius={:.2f}, smallest radius={:.5f}".format(config.in_radius,config.radius))

    # Data augmentation (rotation along the 3 axes).
    # In practice, augmentation only along Z.
    # Do not touch xy. Intuition: no rotation invariance + loss of floor/ceiling orientation (verticality).
    config.x_angle_range = float(args.aug_rot[0])
    config.y_angle_range = float(args.aug_rot[1])
    config.z_angle_range = float(args.aug_rot[2])
    # data augmentation : symetry. In practice, sym. along X and Y
    config.augment_symmetries = [int(float(aus)) for aus in args.aug_sym]
    # data augmentation : scale (random along all axes)
    config.scale_low = 0.7
    config.scale_high = 1.3
    # data augmentation : noise
    config.noise_std = 0.001*config.in_radius*0.5
    config.noise_clip = 0.05*config.in_radius*0.5

    # workers must be equal to the number of cpus.
    config.num_workers = args.num_workers
    # if pretrained network
    config.load_path = args.load_path

    # in number of epochs
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq

    config.rng_seed = args.rng_seed

    config.input_features_dim = 0
    for f in config.features:
        if f=="normal":
            config.input_features_dim += 3
        if "katz" in f:
            config.input_features_dim += len(config.katz_params)
        if f=="intensity":
            config.input_features_dim += 1
    rem = abs(3 - config.input_features_dim%3)%3

    config.input_features_dim += rem

    # GPU rank (if multi-gpu training)
    config.local_rank = args.local_rank

    config.log_dir = os.path.join(args.log_dir, args.job_name)

    config.job_name = args.job_name

    config.TESTED_EPOCH = "BEST"
    config.save_dir = os.path.join(config.log_dir.replace(config.job_name,""), "EVALUATIONS", config.TESTED_EPOCH, config.job_name)
    create_dir_if_required(config.save_dir)

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps: # Number of receptive fields parsed in one epoch. In practice, 2000 is plenty.
        config.num_steps = args.num_steps

    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.lr_decay_steps:
        config.lr_decay_steps = args.lr_decay_steps
    if args.lr_decay_rate:
        config.lr_decay_rate = args.lr_decay_rate

    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    config.from_scratch = args.from_scratch

    config.original_warmup_name = args.original_warmup_name

    return args, config


def get_loader(config):
    # set the data loader
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range),
        d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries),
    ])

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
    ])

    dataset_sampleDl = config.sampleDl
    if "EDF" not in config.dataset:
        type_for_val = config.dataset
    else:
        type_for_val = "other"

    print(config.num_steps)
    print(config.epochs)
    train_dataset = OutlierSegmentationDataset(input_features=config.features, katz_params=config.katz_params, katz_type=config.katz_type,
                             subsampling_parameter=dataset_sampleDl, feature_drop=config.color_drop,
                             in_radius=config.in_radius, num_points=config.num_points,
                             num_steps=config.num_steps, num_epochs=config.epochs,
                             data_root=config.data_root, transforms=train_transforms,
                             split='train',dataset_type=config.dataset, DEBUG=config.DEBUG)
    val_dataset = OutlierSegmentationDataset(input_features=config.features, katz_params=config.katz_params, katz_type=config.katz_type,
                           subsampling_parameter=dataset_sampleDl, feature_drop=config.color_drop,
                           in_radius=config.in_radius, num_points=config.num_points,
                           num_steps=320, num_epochs=1,#config.epochs,
                           data_root=config.data_root, transforms=test_transforms,
                           split='val',dataset_type=type_for_val, DEBUG=config.DEBUG)
    test_dataset = OutlierSegmentationDataset(input_features=config.features, katz_params=config.katz_params, katz_type=config.katz_type,
                           subsampling_parameter=dataset_sampleDl, feature_drop=config.color_drop,
                           in_radius=config.in_radius, num_points=config.num_points,
                           num_steps=config.num_steps, num_epochs=1,
                           data_root=config.data_root, transforms=test_transforms,
                           split='test',dataset_type=config.dataset, DEBUG=config.DEBUG)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=test_sampler,
                                             drop_last=False)

    return train_loader, val_loader, test_loader


def load_checkpoint(config, model, optimizer, scheduler,is_ckpt=True):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    if is_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler, is_best=False):
    logger.info('==> Saving...')
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if is_best:
        torch.save(state, os.path.join(config.log_dir, 'BEST.pth'))
    torch.save(state, os.path.join(config.log_dir, 'current.pth'))
    if epoch % config.save_freq == 0:
        torch.save(state, os.path.join(config.log_dir, 'ckpt_epoch_{}.pth'.format(epoch)))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, 'ckpt_epoch_{}.pth'.format(epoch))))


def main(config):
    train_loader, val_loader, test_loader = get_loader(config)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))
    n_data = len(val_loader.dataset)
    logger.info("length of validation dataset: {}".format(n_data))
    n_data = len(test_loader.dataset)
    logger.info("length of test dataset: {}".format(n_data))

    model, criterion = build_scene_segmentation(config)
    model.cuda()
    criterion.cuda()

    # configuration recommandée : 'sgd'
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.base_learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.base_learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError("Optimizer {} not supported".format(config.optimizer))

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # to have the right LR applied.
    scheduler = get_scheduler(optimizer, len(train_loader), config)

    # resuming from "current.pth" file
    current_ckpt_ls = glob.glob(config.log_dir+"/current.pth")
    config.load_path = None
    config.start_epoch = 0
    if len(current_ckpt_ls)>0 and config.from_scratch==False:
        config.load_path = current_ckpt_ls[0]

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model, optimizer, scheduler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=os.path.join(config.log_dir.replace(config.job_name,""), "TENSORBOARD_SUMMARIES", config.job_name))
    else:
        summary_writer = None

    # save val loss in chkpt
    best_loss = np.Inf
    # routine
    for epoch in range(config.start_epoch, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch - 1
        tic = time.time()
        metrics_train, loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config)

        logger.info('epoch {}, total time {:.2f}, lr {:.5f}'.format(epoch,
                                                                    (time.time() - tic),
                                                                    optimizer.param_groups[0]['lr']))
        is_best = False
        if epoch % config.val_freq == 0:
            metrics_val, loss_val = validate(epoch, val_loader, model, criterion, config)
            if loss_val < best_loss:
                is_best = True
                best_loss = loss_val
                logger.info("Best loss so far: {:.2f} at epoch {:03d}".format(best_loss, epoch))
            if summary_writer is not None:
                for k in SAVED_METRICS:
                    summary_writer.add_scalar('{}_val'.format(k), metrics_val[k], epoch)
                summary_writer.add_scalar('loss_val', loss_val, epoch)

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, epoch, model, optimizer, scheduler, is_best)

        if summary_writer is not None:
            for k in SAVED_METRICS:
                summary_writer.add_scalar('{}_train'.format(k), metrics_train[k], epoch)
            # tensorboard logger
            summary_writer.add_scalar('loss_train', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    final_evaluation(test_loader, model, criterion, config)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    metrics_meter = [AverageMeter()]*len(SAVED_METRICS)

    end = time.time()
    logger.info(len(train_loader.dataset))
    logger.info(len(train_loader))
    for idx, (points, mask, features, points_labels, cloud_label, input_inds,center_ind) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = points.size(0)
        # forward
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)

        pred = model(points, mask, features)

        loss = criterion(pred, points_labels, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()

        metrics = get_metrics_train_and_print(logger.info, config.num_classes, pred.detach().cpu().numpy(), points_labels.int().detach().cpu().numpy(), mask.bool().detach().cpu().numpy(), verbose=(idx % config.print_freq == 0))
        for i, k in enumerate(SAVED_METRICS):
            metrics_meter[i].update(metrics[k],bsz)

        # update meters
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logger.info('Train: [{}/{}][{}/{}]\t'.format(epoch,config.epochs + 1, idx, len(train_loader))+
                        'T {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)+
                        'DT {:.3f} ({:.3f})\t'.format(data_time.val, data_time.avg)+
                        'loss {:.3f} ({:.3f})'.format(loss_meter.val, loss_meter.avg))
            # logger.info(f'[{cloud_label}]: {input_inds}')

    metrics = {}
    for i,k in enumerate(SAVED_METRICS):
        metrics[k] = metrics_meter[i].avg
    return metrics, loss_meter.avg


def validate(epoch, val_loader, model, criterion, config):
    """
    One epoch validating
    """

    batch_time = AverageMeter()
    losses = AverageMeter()

    metrics_meter = [AverageMeter()]*len(SAVED_METRICS)

    model.eval()
    with torch.no_grad():
        end = time.time()

        val_loader.dataset.epoch = 0 # epoch-1 --> SINGLE EPOCH ==> we only use len(val_loader.dataset) patches
        for idx, (points, mask, features, points_labels, cloud_label, input_inds, center_ind) in enumerate(val_loader):
            # forward
            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True)
            cloud_label = cloud_label.cuda(non_blocking=True)
            input_inds = input_inds.cuda(non_blocking=True)

            pred = model(points, mask, features)
            loss = criterion(pred,points_labels,mask)

            bsz = points.shape[0]

            metrics = get_metrics_train_and_print(logger.info, config.num_classes, pred.detach().cpu().numpy(), points_labels.int().detach().cpu().numpy(), mask.bool().detach().cpu().numpy(), verbose=(idx % config.print_freq == 0))
            for i, k in enumerate(SAVED_METRICS):
                metrics_meter[i].update(metrics[k],bsz)
            # update meters
            losses.update(loss.item(), bsz)
            batch_time.update(time.time() - end)
            end = time.time()

    metrics = {}
    for i,k in enumerate(SAVED_METRICS):
        metrics[k] = metrics_meter[i].avg
    return metrics, losses.avg

def final_evaluation(test_loader, model, criterion, config):
    vote_logits_sum = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]
    vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]

    validation_labels = test_loader.dataset.sub_clouds_points_labels

    # check
    val_proportions = np.zeros(config.num_classes, dtype=np.float32)
    for label_value in range(config.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.sub_clouds_points_labels])

    batch_time = AverageMeter()
    losses = AverageMeter()

    num_iterations = len(test_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()

        test_loader.dataset.epoch = 0
        for idx, (points, mask, features, points_labels, cloud_label, input_inds,center_ind) in enumerate(test_loader):
            # forward
            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True)
            cloud_label = cloud_label.cuda(non_blocking=True)
            input_inds = input_inds.cuda(non_blocking=True)

            pred = model(points, mask, features)
            loss = criterion(pred, points_labels, mask)
            losses.update(loss.item(), points.size(0))

            # collect
            bsz = points.shape[0]
            for ib in range(bsz):
                mask_i = mask[ib].cpu().numpy().astype(np.bool)
                logits = pred[ib].cpu().numpy()[:, mask_i]
                inds = input_inds[ib].cpu().numpy()[mask_i]
                c_i = cloud_label[ib].item()
                vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                vote_counts[c_i][:, inds] += 1
                vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx % config.print_freq == 0):
                logger.info("Test iteration {:06d}/{:06d}".format(idx+1,num_iterations))
                metrics = get_metrics_train_and_print(logger.info, config.num_classes, pred.detach().cpu().numpy(), points_labels.int().detach().cpu().numpy(), mask.bool().detach().cpu().numpy(), verbose=1)


        for name,pos,gt,logits,v_count,NN in zip(test_loader.dataset.cloud_names,test_loader.dataset.clouds_points,test_loader.dataset.clouds_points_labels,vote_logits,vote_counts,test_loader.dataset.clouds_points_cluster):
            probas = softmax(logits, axis=0)
            probas_01 = probas[1,:].squeeze()
            count = v_count.squeeze()

            NN = NN.squeeze()
            probas_01 = probas_01[NN]
            count = count[NN]

            write_ply(os.path.join(config.save_dir,name+".ply"),[pos,probas_01,gt,count],["vertex","probas_01","GT","count"])
            np.save(os.path.join(config.save_dir,name+"-probas_01-count.npy"),(probas_01,count))


    compute_metrics(config)


def compute_metrics(config):
    print("Starting metric computation.")

    long_feature_id = []
    if "intensity" in config.features:
        long_feature_id.append("int.")
    if "visibility" in config.features:
        long_feature_id.append("vis.")
    if len(config.features)>0:
        long_feature_id = " + ".join(long_feature_id)
    else:
        long_feature_id = "no features"



    path = config.save_dir

    out_dict = {"Local Aggregator":[],"Features":[],"PR-AUC":[],"macc":[], "miou":[], "prec":[], "rec":[], "fdrate":[], "forate":[], "f_b":[], "TN":[], "FP":[], "FN":[], "TP":[], "Scan ID":[]}
    num_classes = 2

    files = glob.glob(os.path.join(path,"*.npy"))

    target_ls = []
    pred_ls = []
    probas_ls = []

    tic = time.time()
    for file in files:
        filename = os.path.basename(file).replace("-probas_01-count.npy","")

        ply_file = glob.glob(os.path.join(config.data_root,filename+"*.ply"))[0]

        ply = read_ply_ls(ply_file,["vertex","GT"])
        targets = (ply["GT"]==2).astype(np.int32)


        (probas_01,count) = np.load(file)
        preds = probas_01

        C = confusion_matrix(targets, (probas_01>=0.5).astype(np.int32), labels=np.arange(num_classes))
        tn,fp,fn,tp = C.ravel()
        metrics = get_metrics_dict(tn,fp,fn,tp)

        out_dict["Local Aggregator"].append(config.local_aggregator)
        out_dict["Features"].append(long_feature_id)
        out_dict["Scan ID"].append(filename)
        for k in metrics.keys():
            out_dict[k].append(metrics[k])

        P,R,thresh = precision_recall_curve(targets, probas_01)
        pr_auc = auc(R,P)
        out_dict["PR-AUC"].append(100*pr_auc)


        target_ls.append(targets.squeeze())
        pred_ls.append(preds.squeeze())
        probas_ls.append(probas_01.squeeze())


    targets = np.concatenate(target_ls)
    probas = np.concatenate(probas_ls)


    C = confusion_matrix(targets, (probas>=0.5).astype(np.int32), labels=np.arange(num_classes))
    tn,fp,fn,tp = C.ravel()
    metrics = get_metrics_dict(tn,fp,fn,tp)

    out_dict["Local Aggregator"].append(config.local_aggregator)
    out_dict["Features"].append(long_feature_id)
    out_dict["Scan ID"].append("All")
    for k in metrics.keys():
        out_dict[k].append(metrics[k])
    P,R,thresh = precision_recall_curve(targets, probas)
    pr_auc = auc(R,P)
    out_dict["PR-AUC"].append(100*pr_auc)

    out_dict_output_file = r"{}/out_dict_{}_ALL_metrics.pkl".format(config.save_dir,config.job_name)
    with open(out_dict_output_file,"wb") as f:
        pickle.dump(out_dict,f)
    print("All metrics saved. Content:")

    df = pd.DataFrame(out_dict)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir

    print("RANK = {}".format(dist.get_rank()))

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="EDF")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    main(config)
