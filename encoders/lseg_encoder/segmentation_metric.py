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
import re
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

def calculate_accuracy(teacher, student):
    correct_predictions = np.sum(teacher == student)
    total_pixels = np.prod(teacher.shape)
    return correct_predictions / total_pixels

def calculate_accuracy_mask(gt, teacher, student, i):
    mask = np.equal(gt, teacher)
    mask = np.squeeze(mask)
    masked_student = np.where(mask, student, np.nan)
    correct_predictions = np.nansum(masked_student == gt)
    total_pixels = np.sum(mask)
    accuracy = correct_predictions / total_pixels
    return accuracy   
    
def calculate_iou(teacher, student, num_classes):
    iou = []

    unique_labels, counts = np.unique(np.concatenate((teacher, student)), return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]

    for i in sorted_labels[:num_classes]:
        true_labels = teacher == i
        predicted_labels = student == i
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou)  

def calculate_iou_mask(gt, teacher, student, num_classes):
    iou = []

    unique_labels, counts = np.unique(np.concatenate((gt, teacher, student)), return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]


    matching_mask = np.equal(gt, teacher)
    for i in sorted_labels[:num_classes]:
        true_labels = (gt == i) & matching_mask
        predicted_labels = (student == i) & matching_mask
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou) 

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
    if backbone == "fullclip_vitl14_384":
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
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
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
            "--weights", type=str, default=None, help="checkpoint to test"
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
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )
        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
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
        
        ###
        parser.add_argument(
            "--test-rgb-dir",
            help="test rgb dir",
            required=True,
        )

        parser.add_argument(
            "--resize-max", type=float, default=1.25, help=""
        )

        ###
        parser.add_argument(
            "--student-feature-dir",
            help="student feature dir",
            required=True,
        )

        parser.add_argument(
            "--teacher-feature-dir",
            help="teacher feature dir",
            required=True,
        )

        parser.add_argument(
            "--eval-mode",
            help="evaluation mode is either train or test",
            required=True,
        )

        parser.add_argument(
            "--ground-truth",
            help="ground truyh label",
            required=None,
        )
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        return args

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]


###
class FeatureImageFolderLoader(enc_ds.ADE20KSegmentation):#(torch.utils.data.Dataset):
    def __init__(self, student_feature_root, teacher_feature_root, image_root, eval_mode, gt_label_root=None, transform=None):
        self.transform = transform
        self.student_feature_root = student_feature_root
        self.teacher_feature_root = teacher_feature_root
        self.gt_label_root = gt_label_root
        self.image_root = image_root
        self.gt_labels_path = None

        self.student_features = get_folder_features(student_feature_root)
        if eval_mode == 'test':
            self.teacher_features = get_gttest_folder_features(teacher_feature_root)
            if gt_label_root != None:
                self.gt_labels_path = get_gt_label_test(gt_label_root)

        else:
            self.teacher_features = get_folder_features(teacher_feature_root)
            if gt_label_root != None:
                self.gt_labels_path = get_gt_label_train(gt_label_root)
        self.images = get_folder_images(image_root)
        if len(self.student_features) == 0:
            raise(RuntimeError("Found 0 prediction features in subfolders of: \
                " + self.teacher_feature_root + "\n"))
        if len(self.teacher_features) == 0:
            raise(RuntimeError("Found 0 ground truth features in subfolders of: \
                " + self.gt_label_root + "\n"))
        # self.num_class = 150  # ADE20k

    def __getitem__(self, index):
        student_feature = torch.load(self.student_features[index]) 
        teacher_feature = torch.load(self.teacher_features[index])
        if self.gt_labels_path:
            gt_label_path = self.gt_labels_path[index]

            with Image.open(gt_label_path) as img:
                img = np.array(img)
                gt_label = torch.from_numpy(img)
                gt_label = gt_label.unsqueeze(0).unsqueeze(0)
                gt_label = F.interpolate(gt_label, size=(student_feature.shape[1], student_feature.shape[2]), mode='nearest')
                gt_label = gt_label.squeeze(0).squeeze(0)
        else:
            gt_label = None
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return student_feature, teacher_feature, image, os.path.basename(self.student_features[index])

    def __len__(self):
        return len(self.student_features)

###
def get_gttest_folder_features(fea_folder):
    def numerical_sort_key(filename):
        """Extracts numerical part from the filename and returns it for sorting."""
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers]  # returns a list of numbers for multi-level sorting

    glist = glob.glob(fea_folder.rstrip("/") + '/*fmap_CxHxW.pt')
    sorted_list = sorted(glist, key=numerical_sort_key)
    
    # Select every 8th file from the sorted list
    selected_files = sorted_list[2::8]

    return selected_files

def get_folder_features(fea_folder):
    glist = list(glob.glob(fea_folder.rstrip("/") + '/*fmap_CxHxW.pt'))
    return list(sorted(glist))

def get_gt_label_test(fea_folder):
    def numerical_sort_key(filename):
        """Extracts numerical part from the filename and returns it for sorting."""
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers]  # returns a list of numbers for multi-level sorting

    glist = glob.glob(fea_folder.rstrip("/") + '/semantic_class_*.png')
    sorted_list = sorted(glist, key=numerical_sort_key)
    
    # Select every 8th file starting from the 3rd file (index 2) in the sorted list
    selected_files = sorted_list[2::8]

    return selected_files

def get_gt_label_train(fea_folder):
    glist = list(glob.glob(fea_folder.rstrip("/") + '/0*.png'))
    return list(sorted(glist))

def get_folder_images(img_folder):
    glist = list(glob.glob(img_folder.rstrip("/") + '/*.png')) + list(glob.glob(img_folder.rstrip("/") + '/*.jpg'))
    return list(sorted(glist))

def get_feature_image_dataset(student_feature_path, teacher_feature_path, image_path, eval_mode, **kwargs):
    if os.path.isdir(student_feature_path) and os.path.isdir(teacher_feature_path) and os.path.isdir(image_path):
        return FeatureImageFolderLoader(student_feature_path, teacher_feature_path, image_path, eval_mode, transform=kwargs["transform"])

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

    # feature dataset
    testset = get_feature_image_dataset(
        args.student_feature_dir, args.teacher_feature_dir, args.test_rgb_dir, args.eval_mode, transform=input_transform,
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

    """
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        # else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        # else np.linspace(1.0, 3.0, 6)
        #else np.linspace(0.5, 4.0, 6)
        #else np.linspace(0.75, 1.25, 6)
        #else np.linspace(2.0, 4.0, 6)
        #else np.linspace(1.0, 2.0, 6)
        # else np.linspace(0.75, 3.0, 6)
        # else np.linspace(0.75, 2.25, 7)
        else np.linspace(0.75, args.resize_max, 7)
        ## else np.linspace(0.75, 1.75, 7)
        # else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )
    """

    scales = [0.75, 1.0, 1.25, 1.75]
    evaluator = LSeg_MultiEvalModule(
        model, num_classes, scales=scales, flip=True
        ).cuda()
    evaluator.eval()
    tbar = tqdm(test_data)
    w, h = 480, 360
    ################################## encode text feature #################################
    text = ''
    labelset = []
    if args.label_src != 'default':
        labelset = args.label_src.split(',')
    if labelset == []:
        text = clip.tokenize(labels)
    else:
        text = clip.tokenize(args.label_src)
    
    hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }
    clip_pretrained, pretrained = make_encoder(
        args.backbone,
        features=256,
        groups=1,
        expand=False,
        exportable=False,
        hooks=hooks[args.backbone],
        use_readout="project",
        )
    
    text = text.cuda() 
    text_feature = clip_pretrained.encode_text(text) 
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    ########################################################################################
    count = 0
    accuracy_accum = 0
    iou_accum = 0
    for i, (student_features, teacher_features, images, dst) in enumerate(tbar):
        
        """
        if "Replica_Dataset" in args.test_rgb_dir and not (i in train_ids or i in test_ids):
            impath = dst[0]
            print("save dummy array for", impath)
            fmap = np.zeros(1)  # dummy
            np.savez_compressed(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npz".format(w, h)), fmap)
            continue
        """
        count += 1
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
                    )[0] for image in images]
                
                ###
                student_features = [
                    F.interpolate(
                        student_feature.to(torch.float32)[None],
                        size=(h, w),
                        # scale_factor=scale_factor,
                        mode="bilinear",
                        align_corners=True,
                    )[0] for student_feature in student_features] ### [0]

                teacher_features = [
                    F.interpolate(
                        teacher_feature.to(torch.float32)[None],
                        size=(h, w),
                        # scale_factor=scale_factor,
                        mode="bilinear",
                        align_corners=True,
                    )[0] for teacher_feature in teacher_features]
            
            
            student_image_text_features = []
            for student_feature in student_features:
                feashape = student_feature.shape 
                image_feature = student_feature.permute(1, 2, 0).reshape(-1, feashape[0]) 
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).to(torch.float32)


                text_feature = text_feature.to(image_feature.device) 
                logit_scale = logit_scale.to(image_feature.device)
                logits_per_image = image_feature @ text_feature.t() 
                logits_per_image = logits_per_image.view(feashape[1], feashape[2], -1).permute(2, 0, 1)
                pred_image_text_feature = logits_per_image[None] 
                student_image_text_features.append(pred_image_text_feature)

            teacher_image_text_features = []
            for teacher_feature in teacher_features:
                feashape = teacher_feature.shape 
                image_feature = teacher_feature.permute(1, 2, 0).reshape(-1, feashape[0]) # (172800, 512)
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                text_feature = text_feature.to(image_feature.device) # (150, 512)
                logit_scale = logit_scale.to(image_feature.device)
                logits_per_image = image_feature @ text_feature.t() # torch.Size([172800, 150]) logit_scale
                logits_per_image = logits_per_image.view(feashape[1], feashape[2], -1).permute(2, 0, 1)
                gt_image_text_feature = logits_per_image[None] # torch.Size([1, 150, 360, 480])
                teacher_image_text_features.append(gt_image_text_feature)

            teacher_predicts = [
                testset.make_pred(torch.max(pred_image_text_feature, 1)[1].cpu().numpy())
                for pred_image_text_feature in student_image_text_features
            ]
        
            student_predicts = [
                testset.make_pred(torch.max(gt_image_text_feature, 1)[1].cpu().numpy())
                for gt_image_text_feature in teacher_image_text_features
            ]

        
        for teacher_predict, student_predict, impathp in zip(teacher_predicts, student_predicts, dst):
            # prediction and visualize masks
            pred_mask = utils.get_mask_pallete(teacher_predict - 1, 'detail')
            gt_mask = utils.get_mask_pallete(student_predict - 1, 'detail')
            # Visualize accumulated predictions
            pred_mask = torch.tensor(np.array(pred_mask.convert("RGB"), "f")) / 255.0
            gt_mask = torch.tensor(np.array(gt_mask.convert("RGB"), "f")) / 255.0



            gt_mask = gt_mask.cpu().detach().numpy()
            pred_mask = pred_mask.cpu().detach().numpy()

            gtplabel = np.unique(student_predict)
            predpred = np.unique(teacher_predict)
            ###################################### manually change labels here


            for j in range(teacher_predict.shape[1]):
                for k in range(teacher_predict.shape[2]):
                    for element in (teacher_predict, student_predict):
                        # bed sofa cushion pillow = bed
                        if element[0][j][k] == 90:  #TV to door
                            element[0][j][k] = 15
                        if element[0][j][k] == 29:  #rug to floor
                            element[0][j][k] = 4
                        if element[0][j][k] == 58:  #pillow to cushion
                            element[0][j][k] = 40

            #             # floor rug bag building tree= floor
            #             if element[0][j][k] == 29 or element[0][j][k] == 116 or element[0][j][k] == 5:
            #                 element[0][j][k] = 4
            ############## resize to the same size as NeRF
            # student_predict = torch.from_numpy(student_predict).float()
            # student_predict = F.interpolate(student_predict.unsqueeze(0), size=(119, 159), mode='nearest').squeeze(0)
            # student_predict = student_predict.long().numpy()  # Convert back to numpy array

            # teacher_predict = torch.from_numpy(teacher_predict).float()
            # teacher_predict = F.interpolate(teacher_predict.unsqueeze(0), size=(119, 159), mode='nearest').squeeze(0)
            # teacher_predict = teacher_predict.long().numpy()  # Convert back to numpy array


            accuracy = calculate_accuracy(student_predict, teacher_predict)
            # accuracy_mask = calculate_accuracy_mask(gt_label, student_predict, teacher_predict, i)
            iou = calculate_iou(student_predict, teacher_predict, 7)
            # iou_mask = calculate_iou_mask(gt_label, student_predict, teacher_predict, 7)


            accuracy_accum += accuracy
            iou_accum += iou
            print(f"for the {i}th image, the accuracy is {accuracy}, iou is {iou}")
            print(f'the average accuracy is {accuracy_accum/count}, average iou is {iou_accum/count}')
 

def normalize_and_save_images(src_directory, dst_directory_1, dst_directory_2):
    os.makedirs(dst_directory_1, exist_ok=True)
    os.makedirs(dst_directory_2, exist_ok=True)
    png_files = glob.glob(os.path.join(src_directory, '*.png'))
    for file_path in png_files:
        with Image.open(file_path) as img:
            img_array = np.array(img)
            normalized_img_array = img_array * 10
            normalized_img = Image.fromarray((normalized_img_array).astype(np.uint8))
            normalized_img_resize = normalized_img.resize((480, 360))
            base_name = os.path.basename(file_path)

            save_path_1 = os.path.join(dst_directory_1, base_name)
            save_path_2 = os.path.join(dst_directory_2, base_name)
            normalized_img.save(save_path_1)
            normalized_img_resize.save(save_path_2)

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