# visualize Building3D dataset

import os
import yaml
import torch
import argparse
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from Building3D.datasets import build_dataset
from Building3D.datasets.building3d import load_wireframe

import vispy
import vispy.scene as scene

######################################################
DATASET_PATH = './Building3D_entry_level'
CONFIG_PATH = './Building3D/datasets/dataset_config.yaml'
SHARED_CAMERA = True
POINT_SIZE = 3
FOV = 45
BGCOLOR = 'black'
DISTANCE = 5
GT = False
#######################################################

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg

def normalize(vertices, centroid, max_distance):
    return (vertices - centroid) / max_distance

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Building3D dataset')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Path to the dataset')
    parser.add_argument('--config_path', type=str, default=CONFIG_PATH, help='Path to the dataset config file')
    parser.add_argument('--predictions', type=str, default=None, help='Path to the predictions file')
    parser.add_argument('--split', type=str, default='test', help='Split to visualize (train/test)')
    parser.add_argument('--data', type=str, default='tallinn', help='Data type (tallinn/tokyo)')
    args = parser.parse_args()
    return args

def main(dataset_config, predictions_path, split='test', data='tallinn'):
    print('Building3D dataset config:', dataset_config)
    dataset_config['Building3D']['num_points'] = None # TODO: use original point cloud (to be handled in the dataset class)
    dataset_config['Building3D']['normalize'] = True # TODO: use original point cloud (to be handled in the dataset class)

    # setup visualization
    # Create a canvas and add a view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor=BGCOLOR)
    grid = canvas.central_widget.add_grid()

    # Create scatter plot
    view1 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view1, 0, 0)

    view2 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view2, 0, 1)

    markers1 = scene.visuals.Markers()
    markers2 = scene.visuals.Markers()

    # shared camera across all views
    shared_camera = scene.cameras.TurntableCamera(fov=FOV, azimuth=30, distance=DISTANCE)

    # Set view properties
    if SHARED_CAMERA:
        view1.camera = shared_camera
        view2.camera = shared_camera
    else:
        view1.camera = scene.cameras.TurntableCamera(fov=FOV, distance=DISTANCE)
        view2.camera = scene.cameras.TurntableCamera(fov=FOV, distance=DISTANCE)

    view1.add(markers1)
    view2.add(markers2)

    # Initialize the current file index
    index = 0

    # build dataset
    building3d_dataset = build_dataset(dataset_config.Building3D)

    # create dataloader
    data_loader = DataLoader(building3d_dataset[split], batch_size=1, shuffle=False, drop_last=True, num_workers=4, collate_fn=building3d_dataset[split].collate_batch)

    print('Dataset size: ', len(data_loader.dataset))

    # load wireframe predictions
    pred_files = os.listdir(predictions_path)

    # iterator
    iterator = iter(data_loader)

    edge_visual, edge_visual2 = None, None

    def update_view():
        nonlocal edge_visual, edge_visual2
        #clear previous edges
        if edge_visual is not None and edge_visual.parent is not None:
            edge_visual.parent = None
        if edge_visual2 is not None and edge_visual2.parent is not None:
            edge_visual2.parent = None
        # get next batch
        batch = next(iterator)
        pc = batch['points'][0, :, :3].numpy()
        colors = batch['points'][0, :, 3:6].numpy()
        centroid = batch['centroid'][0].numpy() # (3, )
        max_distance = batch['max_distance'][0].numpy() # (1, )
        scan_idx = batch['scan_idx'].item()

        if split == 'train':
            wf_vertices = batch['wf_vertices'][0].numpy()
            wf_edges = batch['wf_edges'][0].numpy().astype(np.int32)

            edges = np.concatenate([[wf_vertices[i], wf_vertices[j]] for (i, j) in wf_edges], axis=0)
            assert len(wf_edges) == len(edges) // 2, f"wf_edges: {len(wf_edges)}, edges: {len(edges)}"      

        #print(f'Point cloud {scan_idx} shape: {pc.shape}')

        # check if vertices are in the point cloud
        # print(wf_vertices[0])
        # distances = np.linalg.norm(pc[:, :3] - wf_vertices[0], axis=1)
        # print(f'Min distance point: {pc[np.argmin(distances)]}')

        # print(f'Points: {len(pc)}')
        # print(wf_vertices.shape, wf_edges.shape)
        # print(wf_vertices)
        # print(wf_edges)

        # predictions
        if data == 'tallinn':
            pred_vertices, pred_edges = load_wireframe(os.path.join(predictions_path, str(scan_idx) + '.obj')) # Building3D dataset
        elif data == 'tokyo':
            pred_vertices, pred_edges = load_wireframe(os.path.join(predictions_path, f'tokyo_{str(scan_idx)}' + '.obj')) # tokyo dataset
        else:
            raise ValueError(f"Unknown data type: {data}")
        # print(f'Predictions: {pred_vertices.shape}, {pred_edges.shape}')
    
        # normalize predictions
        pred_vertices = normalize(pred_vertices, centroid, max_distance)

        print(f'Predicted vertices: {pred_vertices}')
        # print(f'GT vertices: {wf_vertices}')

        pred_edges = np.concatenate([[pred_vertices[i], pred_vertices[j]] for (i, j) in pred_edges], axis=0)

        # assert len(wf_edges) == len(edges), f"wf_edges: {len(wf_edges)}, edges: {len(edges)}"

        # plot also vertices in the point cloud view
        # pc = np.concatenate((pc, wf_vertices), axis=0)
        # colors = np.concatenate((colors, np.ones((len(wf_vertices), 3))), axis=0)

        if GT:
            class_labels = batch['class_label'][0].numpy()
            class_labels = class_labels.astype(bool)
            colors[class_labels] = [1, 0, 0]  # red

        if split == 'train':
            print(f'Point cloud [{scan_idx}] | size {len(pc)} | gt vertices {len(wf_vertices)} | pred vertices {len(pred_vertices)}')
        else:
            print(f'Point cloud [{scan_idx}] | size {len(pc)} | pred vertices {len(pred_vertices)}')


        markers1.set_data(pc, edge_color=colors, face_color=colors, size=POINT_SIZE)
        # add edges to the second view
        if split == 'train':
            markers2.set_data(wf_vertices, edge_color='white', face_color='white', size=POINT_SIZE)
            edge_visual = scene.visuals.Line(edges, connect='segments', color='white', width=POINT_SIZE)
            view2.add(edge_visual)
        # add predictions
        markers2.set_data(pc, edge_color='white', face_color='white', size=POINT_SIZE)
        #markers2.set_data(pred_vertices, edge_color='red', face_color='red', size=POINT_SIZE)
        edge_visual2 = scene.visuals.Line(pred_edges, connect='segments', color='red', width=POINT_SIZE+2)
        view2.add(edge_visual2)

    # Function to handle key press events
    @canvas.events.key_press.connect
    def on_key_press(event):
        # Check if the 'n' key is pressed
        if event.key == 'n':
            # Update the point cloud visualization
            update_view()
        elif event.key == 'q':
            # Close the canvas
            exit()

    update_view()  # Initial update
    # Start the event loop
    vispy.app.run()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    dataset_config = cfg_from_yaml_file(args.config_path)
    dataset_config['Building3D']['root_dir'] = args.dataset_path
    dataset_config['Building3D']['augment'] = False # set to False to avoid augmentations
    #dataset_config['Building3D']['num_points'] = None # use original point cloud

    # predictions_path = os.path.join(args.predictions, 'wireframe')
    predictions_path = args.predictions
    split = args.split
    data = args.data

    main(dataset_config, predictions_path, split, data)