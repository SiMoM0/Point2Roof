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
from test_util import test_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils

# Building3D
from Building3D.datasets import build_dataset
CONFIG_PATH = './Building3D/datasets/dataset_config.yaml'

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--test_tag', type=str, default='pts6', help='extra tag for this experiment')

    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg


def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.test_tag
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    assert output_dir.exists(), '%s does not exist!!!' % str(output_dir)
    ckpt_dir = output_dir / 'ckpt'
    output_dir = output_dir / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(cfg, logger=logger)

    # test_loader = build_dataloader(args.data_path, args.batch_size, cfg.DATA, training=False, logger=logger)
    # Building3D dataset
    dataset_config = cfg_from_yaml_file(CONFIG_PATH)
    dataset_config['Building3D']['root_dir'] = args.data_path
    building3D_dataset = build_dataset(dataset_config.Building3D)
    test_loader = torch.utils.data.DataLoader(
        building3D_dataset['train'], # TODO: change to test
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=building3D_dataset['train'].collate_batch # TODO: change to test
    )
    
    net = RoofNet(cfg.MODEL)
    net.cuda()
    net.eval()

    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        model_utils.load_params(net, ckpt_list[-1], logger=logger)

    logger.info('**********************Start testing**********************')
    logger.info(net)

    test_model(net, test_loader, logger)


if __name__ == '__main__':
    main()
