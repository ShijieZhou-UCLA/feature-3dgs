from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import argparse
import os


parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
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


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        image = cv2.imread(t) # (1423, 1908, 3)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        predictor.set_image(image)
        image_embedding_tensor = torch.tensor(predictor.get_image_embedding().cpu().numpy()[0])
        ###
        img_h, img_w, _ = image.shape
        _, fea_h, fea_w = image_embedding_tensor.shape
        cropped_h = int(fea_w / img_w * img_h + 0.5)
        image_embedding_tensor_cropped = image_embedding_tensor[:, :cropped_h, :]
        print("embedding shape: ", image_embedding_tensor.shape)
        print("image_embedding_tensor_cropped: ", image_embedding_tensor_cropped.shape)
        torch.save(image_embedding_tensor_cropped, os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2


# checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device='cuda')
# predictor = SamPredictor(sam)

# image = cv2.imread("test/images/IMG_20220408_142309.png")
# predictor.set_image(image)
# image_embedding = predictor.get_image_embedding().cpu().numpy()
# print("embedding shape: ", image_embedding.shape)
# np.save("test/embedding.npy", image_embedding)