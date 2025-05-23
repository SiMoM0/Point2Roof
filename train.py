import os
import yaml
import torch
import torch.nn as nn
import argparse
import datetime
import glob
from easydict import EasyDict
import torch.distributed as dist
from dataset.data_utils import build_dataloader
from train_utils import train_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils

# Building3D
from Building3D.datasets import build_dataset
CONFIG_PATH = './Building3D/datasets/dataset_config.yaml'


def get_scheduler(optim, last_epoch):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5, last_epoch=last_epoch)
    return scheduler


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--extra_tag', type=str, default='pts6', help='extra tag for this experiment')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--restart', action='store_true', default=False, help='restart training from the last checkpoint')
    parser.add_argument('--only_edge', action='store_true', default=False, help='train only edge detector')
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)

    return args, cfg

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg

def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.extra_tag if args.extra_tag is not None \
            else 'model-%s' % datetime.datetime.now().strftime('%Y%m%d')
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')

    # Synthetic dataset
    # train_loader = build_dataloader(args.data_path, args.batch_size, cfg.DATA, training=True, logger=logger)
    
    # Building3D dataset
    dataset_config = cfg_from_yaml_file(CONFIG_PATH)
    dataset_config['Building3D']['root_dir'] = args.data_path
    building3D_dataset = build_dataset(dataset_config.Building3D)
    train_loader = torch.utils.data.DataLoader(
        building3D_dataset['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=building3D_dataset['train'].collate_batch
    )

    net = RoofNet(cfg.MODEL)
    net.cuda()

    if args.only_edge:
        net.use_edge = True
        logger.info('Training only edge detector')

        # freeze the keypoint detector and cluster refine net
        for param in net.keypoint_det_net.parameters():
            param.requires_grad = False
        for param in net.cluster_refine_net.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)

    nb_param = sum([p.numel() for p in net.parameters()]) / 1e6
    print(f"Model: {nb_param} x 10^6 trainable parameters ")
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    print(f"Total trainable parameters: {sum([p.numel() for p in trainable_params]) / 1e6} x 10^6")

    # TODO: use a pretrained model
    # last_ckpt = './output/b05/ckpt/checkpoint_epoch_90.pth'
    # if last_ckpt is not None:
    #     logger.info('Loading checkpoint from %s' % last_ckpt)
    #     model_utils.load_params(net, last_ckpt, logger=logger)

    start_epoch = it = 0
    last_epoch = -1
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if args.restart and len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        it, start_epoch = model_utils.load_params_with_optimizer(
            net, ckpt_list[-1], optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1

    scheduler = get_scheduler(optimizer, last_epoch=last_epoch)

    net = net.train()
    logger.info('**********************Start training**********************')
    #logger.info(net)

    train_model(net, optimizer, train_loader, scheduler, it, start_epoch, args.epochs, ckpt_dir)


if __name__ == '__main__':
    main()
