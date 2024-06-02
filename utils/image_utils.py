#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import sklearn.decomposition
import numpy as np
pca_mean = None
top_vector = None
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def feature_map(feature):
    global pca_mean
    global top_vector
    fmap = feature[None, :, :, :]  # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)

    # Reshape and normalize
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3]

    # Perform PCA using torch
    if pca_mean is None:
        pca_mean = f_samples.mean(dim=0, keepdim=True)
    mean = pca_mean
    f_samples_centered = f_samples - mean
    covariance_matrix = f_samples_centered.T @ f_samples_centered / (f_samples_centered.shape[0] - 1)

    eig_values, eig_vectors = torch.linalg.eigh(covariance_matrix)
    if top_vector is None:
        top_vector = eig_vectors[:, -3:]
    top_eig_vectors = top_vector

    transformed = f_samples_centered @ top_eig_vectors

    q1, q99 = transformed.quantile(0.01, dim=0), transformed.quantile(0.99, dim=0)
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = q99 - q1

    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - mean) @ top_eig_vectors
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3))

    return vis_feature.permute(2, 0, 1)  # torch.Size([3, h, w])

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4

    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def depth_to_normal(depth_map, camera):
    # Unproject depth map to obtain 3D points
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    points_world = torch.zeros((height + 1, width + 1, 3)).to(depth_map.device)
    points_world[:height, :width, :] = unproject_depth_map(depth_map, camera)

    # Extract neighboring 3D points
    p1 = points_world[:-1, :-1, :]
    p2 = points_world[1:, :-1, :]
    p3 = points_world[:-1, 1:, :]

    # Compute vectors between neighboring points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute cross product to get normals
    normals = torch.cross(v1, v2, dim=-1)

    # Normalize the normals
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True)+1e-8)

    return normals

def unproject_depth_map(depth_map, camera):
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Reshape the depth map and grid to N x 1
    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Normalize pixel coordinates to [-1, 1]
    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    # Create homogeneous coordinates in the camera space
    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)    

    K_matrix = camera.projection_matrix
    # parse out f1, f2 from K_matrix
    f1 = K_matrix[2, 2]
    f2 = K_matrix[3, 2]

    # get the scaled depth
    sdepth = (f1 * points_camera[..., 2:3] + f2) / (points_camera[..., 2:3] + 1e-8)

    # concatenate xy + scaled depth
    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)
    points_camera = points_camera.view((height,width,3))
    points_camera = torch.cat([points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1)  
    points_world = torch.matmul(points_camera, camera.full_proj_transform.inverse())

    # Discard the homogeneous coordinate
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    points_world = points_world.view((height,width,3))

    return points_world

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'depth':
        net_image = render_pkg["depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'normal':
        net_image = depth_to_normal(render_pkg["depth"], camera).permute(2,0,1)
        net_image = (net_image+1)/2
    elif output == 'curvature':
        net_image = depth_to_normal(render_pkg["depth"], camera).permute(2,0,1)
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    elif output == 'feature map':
        net_image = feature_map(render_pkg['feature_map'])
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image
