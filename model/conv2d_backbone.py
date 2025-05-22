import torch
import torch.nn as nn
import numpy as np
from .pointnet_util import *
from utils import loss_utils

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models import vit_b_16, ViT_B_16_Weights
import segmentation_models_pytorch as smp


"""
# Overlapping patch embedding
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B x C x H' x W'
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B x HW x C
        x = self.norm(x_flat)
        return x, (H, W)

# Transformer encoder block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# SegFormer encoder (MiT-like with 4 stages)
class MiTEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[16, 32, 64, 96], depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 2]):
        super().__init__()
        self.stages = nn.ModuleList()
        input_dim = in_channels
        for i in range(4):
            patch_embed = OverlapPatchEmbed(input_dim, embed_dims[i], stride=4 if i==0 else 2)
            blocks = nn.Sequential(*[TransformerBlock(embed_dims[i], num_heads[i]) for _ in range(depths[i])])
            self.stages.append(nn.Sequential(patch_embed, blocks))
            input_dim = embed_dims[i]

    def forward(self, x):
        features = []
        for stage in self.stages:
            (patch_embed, blocks) = stage
            x, size = patch_embed(x)
            x = blocks(x)
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, *size)
            features.append(x)
        return features  # List of [B, C, H_i, W_i]

# MLP Decoder
class SegFormerDecoder(nn.Module):
    def __init__(self, encoder_dims=[16, 32, 64, 96], out_dim=32, num_classes=21):
        super().__init__()
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for dim in encoder_dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.predict = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, features):
        upsampled = []
        target_size = 128
        for i, feat in enumerate(features):
            feat = self.linear_layers[i](feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)
        fused = self.fuse(torch.cat(upsampled, dim=1))
        out = self.predict(fused)
        return out

# Full SegFormer
class SegFormer(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.encoder = MiTEncoder(in_channels=8)
        self.decoder = SegFormerDecoder(num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # [B, num_classes, H, W]

"""
"""
class TorchvisionViTFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, return_cls=False):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.return_cls = return_cls

    def forward(self, x):
        B = x.size(0)
        H_, W_= x.shape[2], x.shape[3]

        # Step 1: Patchify input
        x = self.vit.conv_proj(x)  # [B, hidden_dim, H', W']
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Step 2: Add CLS token
        cls_token = self.vit.class_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)

        # Step 3: Add positional embedding
        x = x + self.vit.encoder.pos_embedding[:, :(1 + H * W)]
        
        import ipdb; ipdb.set_trace()
        x = self.vit.encoder.dropout(x)  # <--- NOT pos_dropout, it's just 'dropout'


        # Step 4: Transformer
        x = self.vit.encoder(x)  # [B, 1 + N, C]
        x = self.vit.encoder.ln(x)

        if self.return_cls:
            return x[:, 0]  # [B, C] – CLS token
        else:
            patch_tokens = x[:, 1:]  # [B, N, C]
            return patch_tokens.transpose(1, 2).reshape(B, -1, H_, W_)  # [B, C, H', W']
"""
            
class Model2DConv(nn.Module):
    def __init__(self, channels_in):
        super().__init__()

        self.num_output_feature = 32

        self.norm = torch.nn.BatchNorm2d(channels_in)

        # 2d conv backbone
        # self.backbone = SegFormer(num_classes=32)

        # weights = ViT_B_16_Weights.IMAGENET1K_V1
        # self.backbone = vit_b_16(weights=weights)

        # self.backbone = TorchvisionViTFeatureExtractor()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=32)


        self.class_head      = torch.nn.Conv1d(32, 1, kernel_size=1)
        self.regression_head = torch.nn.Conv1d(32, 3, kernel_size=1)

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
            
    def gather_pixel_features(self, img, x_pix, y_pix):
        """
        Gather pixel features at (x_pix, y_pix) for each point in a batch.

        Args:
            img: [B, H, W, C] - CNN features
            x_pix, y_pix: [B, N] - pixel indices per point

        Returns:
            point_features: [B, N, C] - feature per point
        """
        B, H, W, C = img.shape
        _, N = x_pix.shape

        # Flatten the spatial dimensions
        img_flat = img.view(B, -1, C)  # [B, H*W, C]

        # Convert 2D indices to 1D indices
        linear_idx = y_pix * W + x_pix  # [B, N]

        # Create batch offset indices
        batch_indices = torch.arange(B, device=img.device).view(B, 1).expand(B, N)  # [B, N]

        # Gather the features
        point_features = img_flat[batch_indices, linear_idx]  # [B, N, C]

        return point_features

    def forward(self, batch_dict):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""
        if self.training:
            vectors = batch_dict['vectors']
            offset, cls = self.assign_targets(batch_dict["points"], vectors, 0.15) #self.model_cfg.PosRadius)
            self.train_dict.update({
                'offset_label': offset,
                'cls_label': cls
            })

        img = batch_dict['bev_image']
        x_pix, y_pix = batch_dict["inverse_mapping"].permute(1, 0, 2)

        img = img.permute(0, 3, 1, 2).contiguous()

        # print("img shape", img.shape)

        img = self.norm(img)

        # img = self.backbone(img)

        img = self.model(img)

        img = img.permute(0, 2, 3, 1).contiguous()
        # print("img shape", img.shape)

        # pts_feats = img[y_pix, x_pix]
        pts_feats = self.gather_pixel_features(img, x_pix, y_pix)  # [B, N, C]

        # print("pts_feats shape", pts_feats.shape)
        # print("x_pix shape", x_pix.shape)


        pts_feats = pts_feats.permute(0, 2, 1).contiguous()  # [B, C, N]

        class_out = self.class_head(pts_feats).squeeze()
        reg_out = self.regression_head(pts_feats).permute(0, 2, 1).contiguous()  # [B, N, 3]

        if self.training:
            self.train_dict.update({
                'cls_pred': class_out,
                'offset_pred': reg_out
            })
        batch_dict['point_features'] = pts_feats.permute(0, 2, 1) #l0_fea.permute(0, 2, 1)
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