import glob
import tqdm
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
from model.pointnet_util import *
from model.model_utils import *

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write('\n')

def writeEdges(edges, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(edges)):
            edge = edges[i]
            file1.write(str(edge[0] + 1))
            file1.write(' ')
            file1.write(str(edge[1] + 1))
            file1.write(' ')
            file1.write('\n')

def save_wireframe(vertices, edges, wireframe_file):
    r"""
    :param wireframe_file: wireframe file name
    :param vertices: N * 3, vertex coordinates
    :param edges: M * 2,
    :return:
    """
    with open(wireframe_file, 'w') as f:
        for vertex in vertices:
            line = ' '.join(map(str, vertex))
            f.write('v ' + line + '\n')
        for edge in edges:
            edge = ' '.join(map(str, edge + 1))
            f.write('l ' + edge + '\n')

def assign_targets(points, gvs, radius):
    idx = ball_center_query(radius, points, gvs).type(torch.int64)
    batch_size = gvs.size()[0]
    idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
    gvs = gvs.view(-1, 3)
    idx_add += idx
    target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
    dis = target_points - points
    dis[idx < 0] = 0
    dis /= radius
    label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device), torch.zeros(idx.shape).to(idx.device))
    return dis, label

def test_model(model, data_loader, logger, split='train'):
    dataloader_iter = iter(data_loader)
    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        model.use_edge = True
        statistics = {'tp_pts': 0, 'num_label_pts': 0, 'num_pred_pts': 0, 'pts_bias': np.zeros(3, np.float32),
                      'tp_edges': 0, 'num_label_edges': 0, 'num_pred_edges': 0}
        for cur_it in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
            # print(batch.keys())
            # print('GT VERTICES', batch['vectors'].shape, batch['vectors'])
            # print('GT EDGES', batch['edges'].shape, batch['edges'])
            # print(batch['keypoint'].shape, batch['refined_keypoint'].shape)
            # print(batch['keypoint'])
            # print(batch['refined_keypoint'])
            # print(batch['pair_points'].shape)
            # print(batch['pair_points'])
            eval_process(batch, statistics, split)
        bias = statistics['pts_bias'] / statistics['tp_pts']
        logger.info('pts_recall: %f' % (statistics['tp_pts'] / statistics['num_label_pts']))
        logger.info('pts_precision: %f' % (statistics['tp_pts'] / statistics['num_pred_pts']))
        logger.info('pts_bias: %f, %f, %f' % (bias[0], bias[1], bias[2]))
        logger.info('edge_recall: %f' % (statistics['tp_edges'] / statistics['num_label_edges']))
        logger.info('edge_precision: %f' % (statistics['tp_edges'] / statistics['num_pred_edges']))

def eval_process(batch, statistics, split='train'):
    batch_size = batch['batch_size']
    pts_pred, pts_refined = batch['keypoint'], batch['refined_keypoint']
    edge_pred = batch['edge_score']
    if split == 'train':
        pts_label, edge_label = batch['vectors'], batch['edges'] 
    mm_pts = batch['minMaxPt']
    id = batch['frame_id']

    # print(pts_pred.shape, pts_refined.shape, pts_label.shape) # (pred_p, 4), (pred_p, 3), (B, num_p, 3)
    # print(edge_pred.shape, edge_label.shape) # (pred_e, ), (B, num_e, 2)
    # print(pts_pred, pts_refined, pts_label)

    idx = 0
    index = 0
    for i in range(batch_size):
        # print(f'GT VERTICES {pts_label.shape[0]} vs PREDICTED {pts_pred.shape[0]}')
        # print(f'GT EDGES {edge_label.shape[0]} vs PREDICTED {edge_pred.shape[0]}')
        mm_pt = mm_pts[i]
        minPt = mm_pt[0]
        maxPt = mm_pt[1]
        deltaPt = maxPt - minPt

        p_pts = pts_refined[pts_pred[:, 0] == i]
        if split == 'train':
            l_pts = pts_label[i]
            l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]
            vec_a = np.sum(p_pts ** 2, -1)
            vec_b = np.sum(l_pts ** 2, -1)
            dist_matrix = vec_a.reshape(-1, 1) + vec_b.reshape(1, -1) - 2 * np.matmul(p_pts, np.transpose(l_pts))
            dist_matrix = np.sqrt(dist_matrix + 1e-6)
            p_ind, l_ind = linear_sum_assignment(dist_matrix)
            mask = dist_matrix[p_ind, l_ind] < 0.1   # 0.1
            tp_ind, tl_ind = p_ind[mask], l_ind[mask]
            #dis = np.abs(p_pts[tp_ind] - l_pts[tl_ind])
            dis = np.abs( ((p_pts[tp_ind]*deltaPt) + minPt) - ((l_pts[tl_ind]*deltaPt) + minPt) )

            statistics['tp_pts'] += tp_ind.shape[0]
            statistics['num_label_pts'] += l_pts.shape[0]
            statistics['num_pred_pts'] += p_pts.shape[0]
            statistics['pts_bias'] += np.sum(dis, 0)

        # TODO: de-normalize points
        # use centroid and max distance from dataloader

        # print(f'Predicted vertices for frame [{id[i].item()}]:')
        # for idx, pt in enumerate(p_pts_real):
        #     print(f'Point {idx}: {pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}')
        # print(f'Ground truth vertices for frame [{id[i].item()}]:')
        # for idx, pt in enumerate(l_pts):
        #     print(f'Point {idx}: {pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}')

        if split == 'train':
            match_edge = list(itertools.combinations(l_ind, 2))
            match_edge = np.array([tuple(sorted(e)) for e in match_edge])
            score = edge_pred[idx:idx+len(match_edge)]
            idx += len(match_edge)
            l_edge = edge_label[i]
            l_edge = l_edge[np.sum(l_edge, -1, keepdims=False) > 0]
            l_edge = [tuple(e) for e in l_edge]
            match_edge = match_edge[score > 0.5]
            tp_edges = np.sum([tuple(e) in l_edge for e in match_edge])
            statistics['tp_edges'] += tp_edges
            statistics['num_label_edges'] += len(l_edge)
            statistics['num_pred_edges'] += match_edge.shape[0]

        # print(len(l_edge), l_edge)
        # print(match_edge.shape, match_edge)

        # save edges
        all_edges = list(itertools.combinations(range(p_pts.shape[0]), 2))
        all_edges = np.array([tuple(sorted(e)) for e in all_edges])
        num_edges = all_edges.shape[0]
        score = edge_pred[index:index+num_edges]
        index += num_edges
        pred_edges = all_edges[score > 0.5]

        # print(f'Predicted edges for frame [{id[i].item()}]:')
        # for idx, edge in enumerate(pred_edges):
        #     print(f'Edge {idx}: {edge[0]} - {edge[1]}')
        # print(f'Ground truth edges for frame [{id[i].item()}]:')
        # for idx, edge in enumerate(l_edge):
        #     print(f'Edge {idx}: {edge[0]} - {edge[1]}')

        # save wireframe file
        output_dir = os.path.join('./predictions_tokyo/', 'wireframe')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wireframe_file = os.path.join(output_dir, f'{id[i].item()}.obj') # Building3D dataset
        # wireframe_file = os.path.join(output_dir, f'tokyo_{id[i].item()}.obj') # TODO: Tokyo dataset
        save_wireframe(p_pts, pred_edges, wireframe_file)

        # TODO: save points and edges in a .obj file
        # output_dir = os.path.join('./outputs/', 'wireframe')
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # writePoints(p_pts, os.path.join(output_dir, '' + str(id[i].item()) + '.obj'))
        # writeEdges(match_edge, os.path.join(output_dir, '' + str(id[i].item()) + '.edges.obj'))


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        # new code for Building3D
        if not isinstance(val, torch.Tensor):
            continue
        batch_dict[key] = val.cuda(non_blocking=True)
        # old code for synthetic dataset
        # if not isinstance(val, np.ndarray):
            # continue
        # batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_to_cpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        batch_dict[key] = val.cpu().numpy()