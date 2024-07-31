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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
###
import torch.nn.functional as F

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    semantic_feature, ###
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        semantic_feature, ###
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        semantic_feature, ###
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            semantic_feature, ###
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )



        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, feature_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args) #############
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, feature_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args) ###############
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, semantic_feature, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer) ###
        return color, feature_map, radii ###

    @staticmethod
    ### def backward(ctx, grad_out_color, _):
    def backward(ctx, grad_out_color, grad_out_feature, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, semantic_feature, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors ###

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp,
                semantic_feature, ### 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_feature, ###
                sh,
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_semantic_feature, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args) ###
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp,grad_semantic_feature, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args) ###

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp, 
            grad_semantic_feature, ###
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            )
        
        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

# ### ################################################ original #######################################################
# class GaussianRasterizer(nn.Module):
#     def __init__(self, raster_settings):
#         super().__init__()
#         self.raster_settings = raster_settings

#     def markVisible(self, positions):
#         # Mark visible points (based on frustum culling for camera) with a boolean 
#         with torch.no_grad():
#             raster_settings = self.raster_settings
#             visible = _C.mark_visible(
#                 positions,
#                 raster_settings.viewmatrix,
#                 raster_settings.projmatrix)
            
#         return visible
#     ### ADDED FEATURE
#     def forward(self, means3D, means2D, opacities, shs = None, semantic_feature = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
#         raster_settings = self.raster_settings

#         if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
#             raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
#         if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
#             raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
#         if shs is None:
#             shs = torch.Tensor([])
#         if colors_precomp is None:
#             colors_precomp = torch.Tensor([])

#         if scales is None:
#             scales = torch.Tensor([])
#         if rotations is None:
#             rotations = torch.Tensor([])
#         if cov3D_precomp is None:
#             cov3D_precomp = torch.Tensor([])

#         ### original #####################################
#         # Invoke C++/CUDA rasterization routine
#         return rasterize_gaussians(
#             means3D,
#             means2D,
#             shs,
#             colors_precomp,
#             semantic_feature, ###
#             opacities,
#             scales, 
#             rotations,
#             cov3D_precomp,
#             raster_settings,
#         )
# ### ################################################ ^original^ ######################################################







########################################################################################################################
### our CNN feature decoder
########################################################################################################################
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1).cuda()
#         self.bn1 = nn.BatchNorm2d(in_channels).cuda()
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1).cuda()
#         self.bn2 = nn.BatchNorm2d(in_channels).cuda()

#     def forward(self, x):
#         identity = x

#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity  # skip connection
#         return F.relu(out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).cuda()
        self.In1 = nn.InstanceNorm2d(out_channels).cuda()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).cuda()
        self.In2 = nn.InstanceNorm2d(out_channels).cuda()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1).cuda(),
                nn.InstanceNorm2d(out_channels).cuda()
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.In1(self.conv1(x)))
        out = self.In2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        self.layer_sizes = (512, 512, 512, 512, 512) #(64, 128, 256, 512, 512)
        self.input_dim = 512 #32 

        #self.conv5 = nn.Conv2d(self.input_dim, self.layer_sizes[0], kernel_size=5, padding=2).cuda() 
        self.conv5 = nn.Conv2d(self.input_dim, self.layer_sizes[0], kernel_size=1).cuda()
        self.bn5 = nn.InstanceNorm2d(self.layer_sizes[0]).cuda()

        # Residual blocks for channel dimension adjustment
        self.blocks = nn.ModuleList()
        for idx in range(1, len(self.layer_sizes)):
            self.blocks.append(ResidualBlock(self.layer_sizes[idx - 1], self.layer_sizes[idx]))

        # Final convolution to adjust to the desired channel size (512)
        self.final_conv = nn.Conv2d(self.layer_sizes[-1], 512, kernel_size=1).cuda()

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible
    ### ADDED FEATURE
    def forward(self, means3D, means2D, opacities, shs = None, semantic_feature = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        color, feature_map, radii = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            semantic_feature, ###
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
        )

        ### CNN feature decoder for incresing dimemsionality ### 
        x = feature_map.unsqueeze(0)
        # x = F.elu(self.bn5(self.conv5(x)))
        # #print('########################### shape after conv5::::::', x.shape)
        # for layer in self.blocks:
        #     x = layer(x)
        
        x = self.final_conv(x)
        print(self.final_conv.weight)

        feature_map = x.squeeze(0)

        return color, feature_map, radii




