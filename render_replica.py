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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_edit ###
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
###
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral
###
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
###
from utils.clip_utils import CLIPEditor
import yaml
from models.networks import CNN_decoder, MLP_encoder
import time
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr


###
def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    ###
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    # print(pca)
    # print("pca.explained_variance_ratio_", pca.explained_variance_ratio_.tolist())
    # print("pca.singular_values_", pca.singular_values_.tolist())
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    # print(q1, q99)
    del f_samples
    # torch.save({"pca": pca, "feature_pca_mean": feature_pca_mean, "feature_pca_components": feature_pca_components,
    #             "feature_pca_postprocess_sub": feature_pca_postprocess_sub, "feature_pca_postprocess_div": feature_pca_postprocess_div},
    #             os.path.join(outdir, "pca_dict.pt"))
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature


###
def parse_edit_config_and_text_encoding(edit_config):
    edit_dict = {}
    if edit_config is not None:
        with open(edit_config, 'r') as f:
            edit_config = yaml.safe_load(f)
            print(edit_config)
        edit_dict["positive_ids"] = edit_config["query"]["positive_ids"]
        edit_dict["score_threshold"] = edit_config["query"]["score_threshold"]
        if edit_config["query"]["query_type"] == "text":
            # text encoding
            clip_editor = CLIPEditor()
            text_feature = clip_editor.encode_text([t.replace("_", " ") for t in edit_config["query"]["texts"]])
            # print("text_features: ", text_features.shape) # torch.Size([4, 512])
        else:
            raise NotImplementedError

        # setup editing
        op_dict = {}
        for op in edit_config["edit"]["operations"]:
            if op["edit_type"] == "extraction":
                op_dict["extraction"] = True
            elif op["edit_type"] == "deletion":
                op_dict["deletion"] = True
            elif op["edit_type"] == "color_func":
                op_dict["color_func"] = eval(op["func_str"])
            else:
                raise NotImplementedError
        edit_dict["operations"] = op_dict
    return edit_dict, text_feature


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, edit_config, speedup): ###
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map") ###
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map") ###
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature") ###
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration)) ###
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration)) ###
    ### edit
    edit_render_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "renders")
    edit_gts_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "gt")
    edit_feature_map_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "feature_map") ###
    edit_gt_feature_map_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "gt_feature_map") ###
    ### load mlp encoder
    #mlp_encoder = MLP_encoder()
    #mlp_encoder.load_state_dict(torch.load(encoder_ckpt_path))
    ### load cnn decoder
    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

    ### make dirs
    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
        ### edit
        makedirs(edit_render_path, exist_ok=True) ###
        makedirs(edit_gts_path, exist_ok=True) ###
        makedirs(edit_feature_map_path, exist_ok=True) ###
        makedirs(edit_gt_feature_map_path, exist_ok=True) ###
    else:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True) ###
        makedirs(gt_feature_map_path, exist_ok=True) ###
        makedirs(saved_feature_path, exist_ok=True) ###
    acc_psnr = 0
    acc_ssim = 0
    acc_ip = 0
    count = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if edit_config != "no editing":
            render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict) ###
            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() ###
            torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) ###
            torchvision.utils.save_image(gt, os.path.join(edit_gts_path, '{0:05d}'.format(idx) + ".png"))
            ### visualize feature map
            feature_map = render_pkg["feature_map"] ### [None, :, :, :]  # (512, h, w)
            # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        else:
            ### mlp encoder
            #sf_before_render = gaussians.get_semantic_feature #(N,1,512)
            #sf_render = mlp_encoder(sf_before_render) #(N,1,16)
            #gaussians.rewrite_semantic_feature(sf_render)
            ###
            render_pkg = render(view, gaussians, pipeline, background) ###
            #gaussians.rewrite_semantic_feature(sf_before_render)
            count += 1
            gt = view.original_image[0:3, :, :]
            img = render_pkg["render"]
            assim = ssim(gt, img)
            apsnr = psnr(gt, img).mean()
            alpips = lpips(img, gt)
            acc_psnr += apsnr
            acc_ssim += assim
            acc_ip += alpips


            print(f"acc_psnr: {acc_psnr/count}, acc_ssim: {acc_ssim/count}, acc_ip: {acc_ip/count}")

            gt_feature_map = view.semantic_feature.cuda() ###
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) ###
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            ### visualize feature map
            feature_map = render_pkg["feature_map"] ### [None, :, :, :]  # (512, h, w)

            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

            ### save feature map
            ###
            #feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
            ###
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            ##


def render_video(model_path, iteration, views, gaussians, pipeline, background, edit_config): ###
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    render_poses = render_path_spiral(views)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    ###
    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        ###
        if edit_config != "no editing":
            rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) ###
        else:
            rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)

        # print(rendering.max(), rendering.min(), 'aaaa', view.original_image.max(), view.original_image.min())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
        # final_video.write(cv2.imread(os.path.join(render_path, '{0:05d}'.format(idx) + ".png")))
    final_video.release()


def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)


###
def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background, edit_config, speedup): ###
    # non-edit
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map") ###
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map") ###
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature") ###
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration)) ###
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration)) ###

    ### edit
    edit_render_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "renders")
    edit_gts_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "gt")
    edit_feature_map_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "feature_map") ###
    edit_gt_feature_map_path = os.path.join(model_path, name, "ours_edit_{}".format(iteration), "gt_feature_map") ###

    ### load mlp encoder
    #mlp_encoder = MLP_encoder()
    #mlp_encoder.load_state_dict(torch.load(encoder_ckpt_path))
    ### load cnn decoder
    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

    ### make dirs
    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
        makedirs(edit_render_path, exist_ok=True) ###
        makedirs(edit_gts_path, exist_ok=True) ###
        makedirs(edit_feature_map_path, exist_ok=True) ###
        makedirs(edit_gt_feature_map_path, exist_ok=True) ###
    else:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True) ###
        makedirs(gt_feature_map_path, exist_ok=True) ###
        makedirs(saved_feature_path, exist_ok=True) ###

    view = views[0]
    
    ### create novel poses
    render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 200)

    accum_time = 0
    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]          
        
        if edit_config != "no editing":
            render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)
            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() ###
            torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) ###
            torchvision.utils.save_image(gt, os.path.join(edit_gts_path, '{0:05d}'.format(idx) + ".png"))
            ### visualize feature map
            feature_map = render_pkg["feature_map"] ### [None, :, :, :]  # (512, h, w)
            # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        else:
            ### mlp encoder
            #sf_before_render = gaussians.get_semantic_feature #(N,1,512)
            #sf_render = mlp_encoder(sf_before_render) #(N,1,16)
            # gaussians.rewrite_semantic_feature(sf_render)
            ###
            start_time = time.time() ###############
            render_pkg = render(view, gaussians, pipeline, background) ###
       
            # gaussians.rewrite_semantic_feature(sf_before_render)

            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() ###
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) ###


            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            ### visualize feature map
            feature_map = render_pkg["feature_map"] ### [None, :, :, :]  # (512, h, w)
            # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)
            end_time = time.time()
            elapsed_time = end_time - start_time
            accum_time += elapsed_time
            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            # gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            # Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

            ### save feature map
            ###
            #feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
            ###
            # feature_map = feature_map.cpu().numpy().astype(np.float16)
            # torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            ###
    print(f"Process took {accum_time} seconds")



def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, edit_config): ###
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
    
    render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    
    ### create novel poses
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 180)
    
    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        if edit_config != "no editing":
            rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) ###
        else:
            rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


###def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, novel_view : bool, 
                video : bool , edit_config: str, novel_video : bool): ###
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup) ###

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, edit_config, dataset.speedup) ###

        if novel_view: ###
             render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup) ###

        if video:
             render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config) ###

        if novel_video:
             render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup) ###

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true") ###
    parser.add_argument("--novel_video", action="store_true") ###
    parser.add_argument('--edit_config', default="no editing", type=str)
    #parser.add_argument('--edit_config', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.novel_view, args.video, args.edit_config, args.novel_video) ###