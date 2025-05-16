#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py.py
# @IDE     : PyCharm

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist

NUM_NEIGHBORS = 10
NUM_CANDIDATES = 4

def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            if line[0] == 'v':
                vertices.append(line[1:])
            else:
                # if 'line' in line: # debug
                #     print('lineeeeeeeeeeeeeee', line, wireframe_file)
                if 'line' in line or len(line) < 2: # fix dataset error
                    continue
                obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                edges.add(tuple(sorted(obj_data)))
    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(list(edges))
    return vertices, edges


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


def random_sampling(pc, num_points, replace=None, return_choices=False):
    r"""
    :param pc: N * 3
    :param num_points: Int
    :param replace:
    :param return_choices:
    :return:
    """
    if replace is None:
        replace = pc.shape[0] < num_points
    choices = np.random.choice(pc.shape[0], num_points, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        self.normalize = dataset_config.normalize
        self.augment = dataset_config.augment

        assert split_set in ["train", "test"]
        self.split_set = split_set

        self.pc_files, self.wireframe_files = self.load_files()

        if logger:
            logger.info("Total Sample: %d" % len(self.pc_files))

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, index):
        # ------------------------------- Point Clouds ------------------------------
        # load point clouds
        pc_file = self.pc_files[index]
        pc = np.loadtxt(pc_file, dtype=np.float64)

        # point clouds processing
        if not self.use_color:
            point_cloud = pc[:, 0:3]
        elif self.use_color and not self.use_intensity:
            point_cloud = pc[:, 0:7]
            point_cloud[:, 3:] = point_cloud[:, 3:] / 256.0
        elif not self.use_color and self.use_intensity:
            point_cloud = np.concatenate((pc[:, 0:3], pc[:, 7]), axis=1)
        else:
            point_cloud = pc
            point_cloud[:, 3:7] = point_cloud[:, 3:7] / 256.0

        # ------------------------------- Wireframe ------------------------------
        # load wireframe
        if self.split_set == "train":
            wireframe_file = self.wireframe_files[index]
            wf_vertices, wf_edges = load_wireframe(wireframe_file)

        # ------------------------------- Dataset Preprocessing ------------------------------
        if self.normalize:
            centroid = np.mean(point_cloud[:, 0:3], axis=0)
            point_cloud[:, 0:3] -= centroid
            max_distance = np.max(np.linalg.norm(point_cloud[:, 0:3], axis=1))
            point_cloud[:, 0:3] /= max_distance

            if self.split_set == "train":
                wf_vertices -= centroid
                wf_vertices /= max_distance

        if self.num_points:
            point_cloud = random_sampling(point_cloud, self.num_points)

        # neighbor search
        kdtree = KDTree(point_cloud[:, 0:3])
        _, neighbors_emb = kdtree.query(point_cloud[:, 0:3], k=NUM_NEIGHBORS)

        # class labels generation
        # _, cand_vertices = kdtree.query(wf_vertices[:, 0:3], k=NUM_CANDIDATES)
        # cand_vertices = cand_vertices.reshape(-1)
        # assert len(cand_vertices) == NUM_CANDIDATES * wf_vertices.shape[0], \
        #     f"cand_vertices: {len(cand_vertices)}, NUM_CANDIDATES: {NUM_CANDIDATES}, wf_vertices: {wf_vertices.shape[0]}"
        # class_labels = np.zeros((point_cloud.shape[0],), dtype=np.int64)
        # class_labels[cand_vertices] = 1

        # label generation
        # org_distances = cdist(point_cloud[:, :3], wf_vertices[:, :3]) # shape (N, M)
        # old class labels generation using fixed radius
        # distances = np.min(org_distances, axis=1)  # shape (N, ), values in [0, 1]
        # class_labels = (distances < 0.1).astype(np.int64)  # shape (N, ), values in {0, 1}
        # closest_idx = np.argmin(org_distances, axis=1)  # shape (N, ), values in [0, M)
        # offsets = point_cloud[:, 0:3] - wf_vertices[closest_idx, 0:3]  # shape (N, 3)

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                wf_vertices[:, 0] = -1 * wf_vertices[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                wf_vertices[:, 1] = -1 * wf_vertices[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            wf_vertices[:, 0:3] = np.dot(wf_vertices[:, 0:3], np.transpose(rot_mat))

        # minMaxPt for point2roof
        min_pt, max_pt = np.min(point_cloud, axis=0), np.max(point_cloud, axis=0)
        maxXYZ = np.max(max_pt)
        minXYZ = np.min(min_pt)
        min_pt[:] = minXYZ
        max_pt[:] = maxXYZ
        min_pt = min_pt.astype(np.float32)
        max_pt = max_pt.astype(np.float32)
        pt = np.concatenate(( np.expand_dims(min_pt, 0),  np.expand_dims(max_pt, 0)), axis = 0)

        # -------------------------------Edge Vertices ------------------------
        if self.split_set == "train":
            wf_edges_vertices = np.stack((wf_vertices[wf_edges[:, 0]], wf_vertices[wf_edges[:, 1]]), axis=1)
            wf_edges_vertices = wf_edges_vertices[
                np.arange(wf_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(wf_edges_vertices[:, :, -1]),
                                                                            axis=1)]
            wf_centers = (wf_edges_vertices[..., 0, :] + wf_edges_vertices[..., 1, :]) / 2
            wf_edge_number = wf_edges.shape[0]

        # ------------------------------- Return Dict ------------------------------
        ret_dict = {}
        ret_dict['points'] = point_cloud.astype(np.float32)
        if self.split_set == "train":
            ret_dict['vectors'] = wf_vertices.astype(np.float32)
            ret_dict['edges'] = wf_edges.astype(np.int64)
        # ret_dict['wf_centers'] = wf_centers.astype(np.float32)
        # ret_dict['wf_edge_number'] = wf_edge_number
        # ret_dict['wf_edges_vertices'] = wf_edges_vertices.reshape((-1, 6)).astype(np.float32)
        ret_dict['neighbors_emb'] = neighbors_emb.astype(np.int64)
        # ret_dict['class_label'] = class_labels.astype(np.int64)
        # ret_dict['offset'] = offsets.astype(np.float32)
        ret_dict['minMaxPt'] = pt.astype(np.float32)
        if self.normalize:
            ret_dict['centroid'] = centroid
            ret_dict['max_distance'] = max_distance
        idx = os.path.splitext(os.path.basename(pc_file))[0].split('_')[-1]
        ret_dict['frame_id'] = np.array(idx).astype(np.int64)
        return ret_dict

    @staticmethod
    def collate_batch(batch):
        input_dict = defaultdict(list)
        for item in batch:
            for key, val in item.items():
                input_dict[key].append(val)

        ret_dict = {}


        for key, val in input_dict.items():
            try:
                if key in ['vectors', 'edges', 'wf_centers', 'wf_edges_vertices']:
                    max_len = max([len(v) for v in val])
                    wf = np.ones((len(batch), max_len, val[0].shape[-1]), dtype=np.float32) * -1e1
                    for i in range(len(batch)):
                        wf[i, :len(val[i]), :] = val[i]
                    ret_dict[key] = torch.from_numpy(wf)
                else:
                    ret_dict[key] = torch.tensor(np.array(input_dict[key]))
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        
        ret_dict['batch_size'] = len(batch)

        return ret_dict

    def load_files(self):
        data_dir = os.path.join(self.roof_dir, self.split_set)
        pc_files = [pc_file for pc_file in glob.glob(os.path.join(data_dir, 'xyz', '*.xyz'))]
        wireframe_files = [wireframe_file.replace("/xyz", "/wireframe").replace(".xyz", ".obj") for wireframe_file in
                           pc_files]
        return pc_files, wireframe_files

    def print_self_values(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(attribute, "=", value)
