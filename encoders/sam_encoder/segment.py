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
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description=(
        "Segmentation entire image from images with loaded embeddings with no prompt."
    )
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

parser.add_argument(
    "--data",
    type=str,
    help="Path including input images (and features)",
)

parser.add_argument(
    "--iteration",
    type=int,
    required=True,
    help="Chosen number of iterations"
)

parser.add_argument("--image", action="store_true", help="If true, encode feature from image") ###

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


def segment(data_path, name, iteration, from_image, seg_path, mask_generator):
    input_path = os.path.join(data_path, name)
    if not os.path.exists(input_path):
        print(name, "do not exists!")
        return

    # image directory setup
    image_path = os.path.join(input_path, "ours_{}".format(iteration), "renders")
    images = [
            f for f in sorted(os.listdir(image_path)) if not os.path.isdir(os.path.join(image_path, f))
        ]
    images = [os.path.join(image_path, f) for f in images]

    # feature directory setup
    if not from_image:
        feature_path = os.path.join(input_path, "ours_{}".format(args.iteration), "saved_feature")
        features = [
                f for f in sorted(os.listdir(feature_path)) if not os.path.isdir(os.path.join(feature_path, f))
            ]
        features = [os.path.join(feature_path, f) for f in features]
    
    # output directory
    output_path = os.path.join(seg_path, name)
    os.makedirs(output_path, exist_ok=True)

    # get feature list
    image_list = []
    image_name_list = []
    for image in images:
        image_name_list.append(image.split("/")[-1].split(".")[0])
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)

    if not args.image:
        # get image list 
        feature_list = []
        for feature in features:
            feature = torch.load(feature)[None].cuda()
            # add padding to recover the shape (1, 256, 64, 64)
            _, _, fea_h, fea_w = feature.shape
            feature_padding = F.pad(feature, (0, 0, 0, fea_w - fea_h))
            feature_list.append(feature_padding)
        
        # segmentation\
        for i, image in enumerate(tqdm(image_list, desc="Segment progress")):
            masks = mask_generator.generate(image, features=feature_list[i]) ### feature
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plt.show() 
            plt.savefig(os.path.join(output_path, image_name_list[i] + '_seg.png'), 
                        bbox_inches='tight', pad_inches=0)
        
    else:
        # segmentation
        for i, image in enumerate(tqdm(image_list, desc="Segment progress")):
            masks = mask_generator.generate(image) ### feature
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plt.show() 
            plt.savefig(os.path.join(output_path, image_name_list[i] + '_seg.png'), 
                        bbox_inches='tight', pad_inches=0)


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam
    )

    if args.image:
        seg_path = os.path.join(args.data, "seg_{}_img".format(args.iteration))
    else:
        seg_path = os.path.join(args.data, "seg_{}".format(args.iteration))
    os.makedirs(seg_path, exist_ok=True)

    for name in ["test", "train", "novel_views"]:
        segment(args.data, name, args.iteration, args.image, seg_path, mask_generator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
