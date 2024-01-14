import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import warnings
import argparse
import os

from torch.nn import functional as F
import time


parser = argparse.ArgumentParser(
    description=(
        "Segmentation entire image from images with loaded embeddings with no prompt."
    )
)

parser.add_argument(
    "--image_path",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--feature_path",
    type=str,
    help="Path to either a single feature or folder of features.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where semnentation results will be saved. Output will be either a folder "
        "of segmented images or a single segmented image."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam
    )

    # image directory setup
    if not os.path.isdir(args.image_path):
        images = [args.image_path]
    else:
        images = [
            f for f in sorted(os.listdir(args.image_path)) if not os.path.isdir(os.path.join(args.image_path, f))
        ]
        images = [os.path.join(args.image_path, f) for f in images]
    
    # feature directory setup
    if args.feature_path is not None:
        if not os.path.isdir(args.feature_path):
            features = [args.feature_path]
        else:
            features = [
                f for f in sorted(os.listdir(args.feature_path)) if not os.path.isdir(os.path.join(args.feature_path, f))
            ]
            features = [os.path.join(args.feature_path, f) for f in features]

    # output directory
    os.makedirs(args.output, exist_ok=True)

    # get feature list
    image_list = []
    image_name_list = []
    for image in images:
        image_name_list.append(image.split("/")[-1].split(".")[0])
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)

    if args.feature_path is not None:
        # get image list 
        feature_list = []
        for feature in features:
            feature = torch.load(feature)[None].cuda()
            # add padding to recover the shape (1, 256, 64, 64)
            _, _, fea_h, fea_w = feature.shape
            feature_padding = F.pad(feature, (0, 0, 0, fea_w - fea_h))
            feature_list.append(feature_padding)
        
        # segmentation
        print("Start timing - segment from features!")
        start_time = time.time()
        for i, image in enumerate(image_list):
            masks = mask_generator.generate(image, features=feature_list[i]) ### feature
            print("Current speed: ", (i + 1) / (time.time() - start_time))
        end_time = time.time()
        print("Average segment speed from features: ", len(image_list) / (end_time - start_time))
        
    else:
        # segmentation
        print("Start timing - segment from images!")
        start_time = time.time()
        for i, image in enumerate(image_list):
            masks = mask_generator.generate(image) ### feature
            print("Current speed: ", (i + 1) / (time.time() - start_time))
        end_time = time.time()
        print("Average segment speed from features: ", len(image_list) / (end_time - start_time))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
