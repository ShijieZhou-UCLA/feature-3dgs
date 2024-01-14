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
        parser.add_argument(
            "--test-rgb-dir",
            help="test rgb dir",
            required=True,
        )

        parser.add_argument(
            "--resize-max", type=float, default=1.25, help=""
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

    
import matplotlib.patches as mpatches
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
    input_transform = module.val_transform
    # num_classes = module.num_classes
    num_classes = len(labels)
    #labels.append("petal")
    #labels.append("leaf")
    #num_classes += 2

    # dataset
    print("test rgb dir", args.test_rgb_dir)
    testset = get_original_dataset(
        args.test_rgb_dir,
        transform=input_transform,
    )

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
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

    print(model)

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
    # scales = np.linspace(0.75, 2.25, 7)
    # scales = np.linspace(0.75, args.resize_max, 7)
    # scales = [0.75, 1.0, 1.25]
    # scales = [0.75, 1.0, 1.25, 1.5, 1.75]
    scales = [0.75, 1.0, 1.25, 1.75]
    print("scales", scales)
    print("test rgb dir", args.test_rgb_dir)
    print("outdir", args.outdir)
    evaluator = LSeg_MultiEvalModule(
        model, num_classes, scales=scales, flip=True
    ).cuda()
    evaluator.eval()

    metric = utils.SegmentationMetric(testset.num_class)
    tbar = tqdm(test_data)

    f = open("log_test_{}_{}.txt".format(args.jobname, args.dataset), "a+")
    per_class_iou = np.zeros(testset.num_class)
    print(testset.num_class)
    cnt = 0

    if "Replica_Dataset" in args.test_rgb_dir:
        print(args.data_path, "is Replica_Dataset. So, skip some frames.")
        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))
        test_ids = [x+step//2 for x in train_ids]
        assert len(testset) == total_num, (len(testset), total_num)
        assert args.test_batch_size == 1, args.test_batch_size

    # output folder
    # outdir = "outdir_ours"
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if "Replica_Dataset" in args.test_rgb_dir:
        w, h = 320, 240
    elif "scannnet" in args.test_rgb_dir:
        w, h = 320, 240
    else:
        w, h = 480, 360
    print("w, h =", w, h)

    pca = None
    print("scales", scales)
    print("test rgb dir", args.test_rgb_dir)
    print("outdir", args.outdir)
    for i, (image, dst) in enumerate(tbar):
        """
        if "Replica_Dataset" in args.test_rgb_dir and not (i in train_ids or i in test_ids):
            impath = dst[0]
            print("save dummy array for", impath)
            fmap = np.zeros(1)  # dummy
            np.savez_compressed(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npz".format(w, h)), fmap)
            continue
        """

        with torch.no_grad():
            print(image[0].shape, "image.shape -")
            # print([im.shape for im in image]) # [torch.Size([3, 1438, 1918])]
            #if image[0].shape[-1] > 1008:
            #    scale_factor = 1008 / image[0].shape[-1]
            if image[0].shape[-1] > w:
                # scale_factor = w / image[0].shape[-1]
                # print("resize", image[0].shape, "to", [int(s * scale_factor) for s in image[0].shape])
                print("resize", image[0].shape, "to", (h, w))
                image = [
                    F.interpolate(
                        img[None],
                        size=(h, w),
                        # scale_factor=scale_factor,
                        mode="bilinear",
                        align_corners=True,
                    )[0] for img in image]
                print(image[0].shape)
            outputs = evaluator.parallel_forward(image)
            print(image[0].shape, "image.shape")
            print("start pred")
            start = time.time()
            output_features = evaluator.parallel_forward(image, return_feature=True)
            # print(output_features.shape, output_features.min(), output_features.max())
            # print(type(outputs), type(output_features))
            print("done pred", start - time.time())
            # list
            print("start make_pred")
            start = time.time()
            predicts = [
                testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                for output in outputs
            ]
            print("done makepred", start - time.time())
            # output_features = [o.cpu().numpy().astype(np.float16) for o in output_features]

        for predict, impath, img, fmap in zip(predicts, dst, image, output_features):
            # prediction mask
            # mask = utils.get_mask_pallete(predict - 1, args.dataset)
            mask = utils.get_mask_pallete(predict - 1, 'detail')
            outname = os.path.splitext(impath)[0] + ".png"
            mask.save(os.path.join(outdir, outname))

            # vis from accumulation of prediction
            mask = torch.tensor(np.array(mask.convert("RGB"), "f")) / 255.0
            vis_img = (img + 1) / 2.
            vis_img = vis_img.permute(1, 2, 0)  # ->hwc
            vis1 = vis_img
            vis2 = vis_img * 0.4 + mask * 0.6
            vis3 = mask
            vis = torch.cat([vis1, vis2, vis3], dim=1)
            Image.fromarray((vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(outdir, outname + "_vis.png"))

            # new_palette = get_new_pallete(len(labels))
            # seg, patches = get_new_mask_pallete(predict, new_palette, labels=labels)
            # print(predict.min(), predict.max())
            seg, patches = get_legend_patch(predict - 1, adepallete, labels)
            seg = seg.convert("RGBA")
            plt.figure()
            plt.axis('off')
            plt.imshow(seg)
            #plt.legend(handles=patches)
            plt.legend(handles=patches, prop={'size': 8}, ncol=4)
            plt.savefig(os.path.join(outdir, outname + "_legend.png"), format="png", dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

            ###############################
            # print(fmap.shape)  # torch.Size([1, 512, 512, 683])
            #print(fmap.shape, h, w)
            start = time.time()
            fmap = F.interpolate(fmap, size=(h, w), mode='bilinear', align_corners=False)  # [1, 512, h, w]
            fmap = F.normalize(fmap, dim=1)  # normalize
            #print(time.time() - start)
            #print("done interpolate")

            if pca is None:
                print("calculate PCA based on 1st image", impath)
                pca = sklearn.decomposition.PCA(3, random_state=42)
                f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
                transformed = pca.fit_transform(f_samples)
                print(pca)
                print("pca.explained_variance_ratio_", pca.explained_variance_ratio_.tolist())
                print("pca.singular_values_", pca.singular_values_.tolist())
                feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
                feature_pca_components = torch.tensor(pca.components_).float().cuda()
                q1, q99 = np.percentile(transformed, [1, 99])
                feature_pca_postprocess_sub = q1
                feature_pca_postprocess_div = (q99 - q1)
                print(q1, q99)
                del f_samples
                torch.save({"pca": pca, "feature_pca_mean": feature_pca_mean, "feature_pca_components": feature_pca_components,
                            "feature_pca_postprocess_sub": feature_pca_postprocess_sub, "feature_pca_postprocess_div": feature_pca_postprocess_div},
                           os.path.join(outdir, "pca_dict.pt"))

            #print("start imgsave")
            start = time.time()
            vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
            Image.fromarray((vis_feature.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(outdir, outname + "_feature_vis.png"))
            #print(time.time() - start)
            #print("done imgsave")

            fmap = fmap[0]  # [512, h, w]
            fmap = fmap.cpu().numpy().astype(np.float16)
            # np.save(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npy".format(w, h)), fmap)
            #print("start savez")
            #start = time.time()
            #np.savez_compressed(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npz".format(w, h)), fmap)  # 70% filesize
            #print(time.time() - start)
            #print("done savez")
            print("start save")
            start = time.time()
            # np.save(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npy".format(w, h)), fmap)
            print(fmap.shape)
            torch.save(torch.tensor(fmap).half(), os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap_CxHxW.pt"))
            print(time.time() - start)
            # print("done save")
 


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
