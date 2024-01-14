import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet
from modules.lseg_module import LSegModule
#from utils import Resize
from transforms_midas import Resize
import cv2
import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data import get_dataset, get_original_dataset
from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
import torchvision.transforms as transforms
import sklearn
import sklearn.decomposition
import time

import glob
import encoding.datasets as enc_ds
import torch.nn as nn

from modules.models.lseg_vit import (
    _make_pretrained_clip_vitl16_384,
    _make_pretrained_clip_vitb32_384,
    _make_pretrained_clipRN50x16_vitl16_384,
    forward_vit,
    _make_fullclip_vitl14_384,
)


def make_encoder(
    backbone,
    features=256,
    use_pretrained=True,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):
    if backbone == "fullclip_vitl14_384": # full clip
        clip_pretrained, pretrained = _make_fullclip_vitl14_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "clip_vitl16_384": 
        clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "clipRN50x16_vitl16_384":
        clip_pretrained, pretrained = _make_pretrained_clipRN50x16_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "clip_vitb32_384":
        clip_pretrained, pretrained = _make_pretrained_clip_vitb32_384(
            use_pretrained, 
            hooks=hooks, 
            use_readout=use_readout,
        )
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return clip_pretrained, pretrained


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            # default="ade20k",
            default="ignore",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument( 
            "--workers", type=int, default=0, metavar="N", help="dataloader threads" ### default=16
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--weights", type=str, default="demo_e200.ckpt", help="checkpoint to test" ### default=None
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )
        # test option
        parser.add_argument(
            "--data-path", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=False,
            action="store_false",
            help="turn off scaleinv layers" ### default=True
        )
        parser.add_argument(
            "--widehead", default=True, action="store_true", help="wider output head" ### default=False
        )
        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )
        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )
        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )
        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )
        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        parser.add_argument(
            "--outdir",
            default="outdir_ours",
            help="output direcory of features",
        )

        parser.add_argument(
            "--resize-max", type=float, default=1.25, help=""
        )

        ###
        parser.add_argument(
            "--data",
            type=str,
            help="Path including input images (and features)",
        )

        ###
        parser.add_argument(
            "--iteration",
            type=int,
            required=True,
            help="Chosen number of iterations"
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,
              230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,
              4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,
              220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,
              8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,
              163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,
              235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,
              163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,
              153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,
              255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,
              0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,
              255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,
              255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,
              245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,
              255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,
              194,194,102,255,0,92,0,255]


###
class FeatureImageFolderLoader(enc_ds.ADE20KSegmentation):#(torch.utils.data.Dataset):
    def __init__(self, feature_root, image_root, transform=None):
        self.transform = transform
        self.feature_root = feature_root
        self.image_root = image_root
        self.features = get_folder_features(feature_root)
        self.images = get_folder_images(image_root)
        if len(self.features) == 0:
            raise(RuntimeError("Found 0 features in subfolders of: \
                " + self.root + "\n"))
        # self.num_class = 150  # ADE20k

    def __getitem__(self, index):
        feature = torch.load(self.features[index]) ### [0]
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return feature, image, os.path.basename(self.features[index])

    def __len__(self):
        return len(self.features)


def get_folder_features(fea_folder):
    glist = list(glob.glob(fea_folder.rstrip("/") + '/*_fmap_CxHxW.pt'))
    return list(sorted(glist))


def get_folder_images(img_folder):
    glist = list(glob.glob(img_folder.rstrip("/") + '/*.png')) + list(glob.glob(img_folder.rstrip("/") + '/*.jpg'))
    return list(sorted(glist))


def get_feature_image_dataset(feature_path, image_path, **kwargs):
    if os.path.isdir(feature_path) and os.path.isdir(image_path):
        return FeatureImageFolderLoader(feature_path, image_path, transform=kwargs["transform"])


def get_legend_patch(npimg, new_palette, labels):
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)
    u_index = np.unique(npimg)
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    return out_img, patches


def test(args):

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        strict=args.strict,
        block_depth=args.block_depth,
        activation=args.activation,
    )
    labels = module.get_labels('ade20k')
    num_classes = len(labels)
    input_transform = module.val_transform

    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )


    # encode text feature
    text = ''
    labelset = []
    if args.label_src != 'default':
        labelset = args.label_src.split(',')
    if labelset == []:
        text = clip.tokenize(labels)
    else:
        text = clip.tokenize(labelset)
    
    hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }
    clip_pretrained, _ = make_encoder(
        args.backbone,
        features=256,
        groups=1,
        expand=False,
        exportable=False,
        hooks=hooks[args.backbone],
        use_readout="project",
        )
    
    text = text.cuda() # text = text.to(x.device) # TODO: need use correct device
    text_feature = clip_pretrained.encode_text(text) # torch.Size([150, 512])
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    # logit_scale = logit_scale.cuda() # logit_scale = logit_scale.to(x.device) # TODO: need use correct device


    if labelset == []:
        seg_path = os.path.join(args.data, "seg_{}_labels".format(args.iteration))
    else:
        seg_path = os.path.join(args.data, "seg_{}".format(args.iteration))
    os.makedirs(seg_path, exist_ok=True)

    for name in ["test", "train", "novel_views"]:
        input_path = os.path.join(args.data, name)
        if not os.path.exists(input_path):
            print(name, "do not exists!")
            continue
        image_path = os.path.join(input_path, "ours_{}".format(args.iteration), "renders")
        feature_path = os.path.join(input_path, "ours_{}".format(args.iteration), "saved_feature")
        
        output_path = os.path.join(seg_path, name)
        os.makedirs(output_path, exist_ok=True)


        # feature dataset
        testset = get_feature_image_dataset(
            feature_path, image_path, transform=input_transform,
        )
        test_data = data.DataLoader(
            testset,
            batch_size=args.test_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=test_batchify_fn,
            **loader_kwargs
        )

        if isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module

        model = model.eval()
        model = model.cpu()

        if args.export:
            torch.save(model.state_dict(), args.export + ".pth")
            return

        scales = [0.75, 1.0, 1.25, 1.75]

        evaluator = LSeg_MultiEvalModule(
            model, num_classes, scales=scales, flip=True
        ).cuda()
        evaluator.eval()

        tbar = tqdm(test_data)
        
        w, h = 480, 360

        for _, (features, images, dst) in enumerate(tbar):
            with torch.no_grad():
                if images[0].shape[-1] > w:
                    print("resize", images[0].shape, "to", (h, w))

                    images = [
                        F.interpolate(
                            image[None],
                            size=(h, w),
                            # scale_factor=scale_factor,
                            mode="bilinear",
                            align_corners=True,
                        )[0] for image in images] # images[0]: torch.Size([3, 1440, 1920])
                    
                    ###
                    features = [
                        F.interpolate(
                            feature.to(torch.float32)[None],
                            size=(h, w),
                            # scale_factor=scale_factor,
                            mode="bilinear",
                            align_corners=True,
                        )[0] for feature in features] ### features[0]: torch.Size([512, 1440, 1920])
                
                ###
                image_text_features = []
                for feature in features:
                    feashape = feature.shape # (512, 360, 480)
                    # feature = F.normalize(feature.float(), dim=0).half()
                    image_feature = feature.permute(1, 2, 0).reshape(-1, feashape[0]) # (172800, 512)
                    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                    text_feature = text_feature.to(image_feature.device) # (150, 512)
                    logit_scale = logit_scale.to(image_feature.device)
                    logits_per_image = image_feature @ text_feature.t() # torch.Size([172800, 150]) logit_scale
                    logits_per_image = logits_per_image.view(feashape[1], feashape[2], -1).permute(2, 0, 1) # torch.Size([150, 360, 480])
                    image_text_feature = logits_per_image[None] # torch.Size([1, 150, 360, 480])
                    image_text_features.append(image_text_feature)


                predicts = [
                    testset.make_pred(torch.max(image_text_feature, 1)[1].cpu().numpy())
                    for image_text_feature in image_text_features
                ]
            
            # fmap: torch.Size([1, 512, 1194, 1600]), img: torch.Size([3, 360, 480])
            for predict, impath, img, fmap in zip(predicts, dst, images, features):
                # prediction and visualize masks
                mask = utils.get_mask_pallete(predict - 1, 'detail')
                outname = os.path.splitext(impath)[0] + ".png"
                mask.save(os.path.join(output_path, outname))

                # Visualize accumulated predictions
                mask = torch.tensor(np.array(mask.convert("RGB"), "f")) / 255.0
                vis_img = (img + 1) / 2.
                vis_img = vis_img.permute(1, 2, 0)  # ->hwc
                vis1 = vis_img
                vis2 = vis_img * 0.4 + mask * 0.6
                vis3 = mask
                vis = torch.cat([vis1, vis2, vis3], dim=1)
                Image.fromarray((vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(output_path, outname + "_vis.png"))

                # Visualize results with legend
                if labelset == []:
                    seg, patches = get_legend_patch(predict - 1, adepallete, labels)
                else:
                    seg, patches = get_legend_patch(predict - 1, adepallete, labelset)
                
                seg = seg.convert("RGBA")
                plt.figure()
                plt.axis('off')
                plt.imshow(seg)
                #plt.legend(handles=patches)
                plt.legend(handles=patches, prop={'size': 8}, ncol=4)
                plt.savefig(os.path.join(output_path, outname + "_legend.png"), format="png", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
    


class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count() 
    test(args)
