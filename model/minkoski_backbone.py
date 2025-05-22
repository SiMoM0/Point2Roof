import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from .backbone import MinkEncoderDecoder
from .pointnet_util import *
from utils import loss_utils


def TensorField(x, f):
    """
    Build a tensor field from coordinates and features in the 
    input batch
    The coordinates are quantized using the provided resolution

    """
    features=torch.from_numpy(np.concatenate(f, axis=0)).float()
    features = features.reshape(features.shape[0], -1)
    feat_tfield = ME.TensorField(
        features=features,
        coordinates=x,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device="cuda",
    )
    return feat_tfield


res = 0.1

class ModelTwo(nn.Module):
    def __init__(self, channels_in):
        super().__init__()


        self.backbone = MinkEncoderDecoder()

        self.num_output_feature = 32

        self.norm = ME.MinkowskiBatchNorm(channels_in)

        self.class_head      = ME.MinkowskiConvolution(32, 1, kernel_size=1, dimension=3)
        self.regression_head = ME.MinkowskiConvolution(32, 3, kernel_size=1, dimension=3)

        if self.training:
            self.train_dict = {}
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidBCELoss()
            )
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss()
            )
            self.loss_weight = {
                                    'cls_weight': 1.0,
                                    'reg_weight': 1.0
                                    } #self.model_cfg.LossWeight

    def forward(self, batch_dict):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""

        xyz = batch_dict['points'][:, :, :3]

        B, N, _ = xyz.shape

        coords = ME.utils.batched_coordinates([i / res for i in xyz], dtype=torch.float32)

        if self.training:
            vectors = batch_dict['vectors']
            offset, cls = self.assign_targets(xyz, vectors, 0.15) #self.model_cfg.PosRadius)
            self.train_dict.update({
                'offset_label': offset,
                'cls_label': cls
            })

        point_features = batch_dict['points'].reshape(B*N, -1) #np.creg_outoncatenate(xyz, axis=0)

        t_field = ME.TensorField(
            features=point_features,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device="cuda",
        )
        x = self.norm(t_field.sparse())

        x = self.backbone(x)

        # Classification
        class_out = self.class_head(x)
        class_out = class_out.slice(t_field)
        class_out = class_out.F.reshape(B, N, 1) #torch.cat(class_out.decomposed_features, dim=0)

        reg_feats = x.slice(t_field)
        reg_feats = reg_feats.F.reshape(B, N, 32) #reg_out.decomposed_features #torch.cat(reg_out.decomposed_features, dim=0)

        # Regression
        reg_out = self.regression_head(x)
        reg_out = reg_out.slice(t_field)
        reg_out = reg_out.F.reshape(B, N, 3) #reg_out.decomposed_features #torch.cat(reg_out.decomposed_features, dim=0)

        if self.training:
            self.train_dict.update({
                'cls_pred': class_out,
                'offset_pred': reg_out
            })
        batch_dict['point_features'] = reg_feats #l0_fea.permute(0, 2, 1)
        batch_dict['point_pred_score'] = torch.sigmoid(class_out).squeeze(-1)
        batch_dict['point_pred_offset'] = reg_out * 0.15 #self.model_cfg.PosRadius  
        

        return batch_dict
    
    def loss(self, loss_dict, disp_dict):
        pred_cls, pred_offset = self.train_dict['cls_pred'], self.train_dict['offset_pred']
        label_cls, label_offset = self.train_dict['cls_label'], self.train_dict['offset_label']
        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        reg_loss = self.get_reg_loss(pred_offset, label_offset, label_cls, self.loss_weight['reg_weight'])
        loss = cls_loss + reg_loss
        loss_dict.update({
            'pts_cls_loss': cls_loss.item(),
            'pts_offset_loss': reg_loss.item(),
            'pts_loss': loss.item()
        })

        pred_cls = pred_cls.squeeze(-1)
        label_cls = label_cls.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls)
        pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape), pred_logit.new_zeros(pred_logit.shape))
        acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / torch.sum(label_cls == 1).item()
        #acc = torch.sum(pred == label_cls).item() / len(label_cls.view(-1))
        disp_dict.update({'pts_acc': acc})
        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        batch_size = int(pred.shape[0])
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * weight
        return cls_loss

    def get_reg_loss(self, pred, label, cls_label, weight):
        batch_size = int(pred.shape[0])
        positives = cls_label > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_loss_src = self.reg_loss_func(pred, label, weights=reg_weights)  # [N, M]
        reg_loss = reg_loss_src.sum() / batch_size
        reg_loss = reg_loss * weight
        return reg_loss

    def assign_targets(self, points, gvs, radius):
        idx = ball_center_query(radius, points[:, :, :3], gvs).type(torch.int64)
        batch_size = gvs.size()[0]
        idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
        gvs = gvs.view(-1, 3)
        idx_add += idx
        target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
        dis = target_points - points[:, :, :3]
        dis[idx < 0] = 0
        dis /= radius
        label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device),
                            torch.zeros(idx.shape).to(idx.device))
        return dis, label