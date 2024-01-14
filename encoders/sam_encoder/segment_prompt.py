import torch
import numpy as np
import cv2


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

import warnings
import argparse
import os

from torch.nn import functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description=(
        "Segment images with prompt."
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
    "--onnx_path",
    type=str,
    default='sam_onnx_example.onnx',
    # required=True,
    help="onnx model path",
)

parser.add_argument(
    "--onnx_quantized_path",
    type=str,
    default='',
    # required=True,
    help="onnx model quantized path",
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

parser.add_argument('--point', 
                    type=int,
                    nargs='+',
                    help='two values x, y as a input point')

parser.add_argument('--box', 
                    type=int,
                    nargs='+',
                    help='four values x1, y1 as top left and x2, y2 corner as a bottom right corner')


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    # onnx model setup
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(args.onnx_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version= 12,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )    
    
    if args.onnx_quantized_path != '':
        quantize_dynamic(
            model_input=args.onnx_path,
            model_output=args.onnx_quantized_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        args.onnx_path = args.onnx_quantized_path
    
    # # using an ONNX model
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)


    # make seg dirs
    if not args.image:
        if args.box is not None and args.point is None:
            seg_path = os.path.join(args.data, "seg_box_{}".format(args.iteration))
        elif args.box is None and args.point is not None:
            seg_path = os.path.join(args.data, "seg_point_{}".format(args.iteration))
        elif args.box is not None and args.point is not None:
            seg_path = os.path.join(args.data, "seg_box_point_{}".format(args.iteration))
    else:
        if args.box is not None and args.point is None:
            seg_path = os.path.join(args.data, "seg_box_{}_img".format(args.iteration))
        elif args.box is None and args.point is not None:
            seg_path = os.path.join(args.data, "seg_point_{}_img".format(args.iteration))
        elif args.box is not None and args.point is not None:
            seg_path = os.path.join(args.data, "seg_box_point_{}_img".format(args.iteration))
    os.makedirs(seg_path, exist_ok=True)


    for name in ["test", "train", "novel_views"]:
        input_path = os.path.join(args.data, name)
        if not os.path.exists(input_path):
            print(name, "do not exists!")
            continue
        
        # image directory setup
        image_path = os.path.join(input_path, "ours_{}".format(args.iteration), "renders")
        images = [
                f for f in sorted(os.listdir(image_path)) if not os.path.isdir(os.path.join(image_path, f))
            ]
        images = [os.path.join(image_path, f) for f in images]

        # feature directory setup
        if not args.image:
            feature_path = os.path.join(input_path, "ours_{}".format(args.iteration), "saved_feature")
            features = [
                    f for f in sorted(os.listdir(feature_path)) if not os.path.isdir(os.path.join(feature_path, f))
                ]
            features = [os.path.join(feature_path, f) for f in features]

    
        # output directory
        output_path = os.path.join(seg_path, name)
        os.makedirs(output_path, exist_ok=True)


        # get image size
        img = cv2.imread(images[0])
        if args.point is None and args.box is not None: # only box input
            # Input box
            input_box = np.array(args.box)
            # Add a batch index, concatenate a padding point, and transform.
            onnx_box_coords = input_box.reshape(2, 2)
            onnx_box_labels = np.array([2,3])
            onnx_coord = onnx_box_coords[None, :, :]
            onnx_label = onnx_box_labels[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)
        elif args.point is not None and args.box is None: # only point input
            # Input point
            input_point = np.array([args.point])
            input_label = np.array([1])
            # Add a batch index, concatenate a padding point, and transform.
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)
        elif args.point is not None and args.box is not None: # both box and point input
            input_box = np.array(args.box)
            input_point = np.array([args.point])
            input_label = np.array([0])
            onnx_box_coords = input_box.reshape(2, 2)
            onnx_box_labels = np.array([2,3])
            onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)
        else: # no box and no input
            onnx_coord = np.array([[0.0, 0.0], [0.0, 0.0]])[None, :, :]
            onnx_label = np.array([-1,-1])[None, :].astype(np.float32)
            print("point: ", onnx_coord)
            print("box: ", onnx_label)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)


        for i, image in enumerate(tqdm(images, desc="Segment progress")):
            image_name = image.split("/")[-1].split(".")[0]
            print(f"Processing '{image}' ...")
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (356, 477, 3)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.axis('on')
            plt.show()

            predictor.set_image(image)
            

            if args.image:
                embedding = predictor.get_image_embedding().cpu().numpy()
            else:
                embedding = torch.load(features[i])[None] # (1, 256, 48, 64)
                # add padding to recover the shape (1, 256, 64, 64)
                _, _, fea_h, fea_w = embedding.shape
                embedding = F.pad(embedding, (0, 0, 0, fea_w - fea_h))
                embedding = embedding.cpu().numpy().astype(np.float32) # (1, 256, 64, 64)

            

            # Package the inputs to run in the onnx model
            ort_inputs = {
                "image_embeddings": embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
            }
            # Predict a mask and threshold it.
            masks, _, low_res_logits = ort_session.run(None, ort_inputs)
            masks = masks > predictor.model.mask_threshold
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # show mask
            show_mask(masks[0], plt.gca())
            # show box
            if args.box is not None:
                show_box(input_box, plt.gca())
            # show point
            if args.point is not None:
                show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(output_path, image_name + '.png'), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# import cv2
# from matplotlib import pyplot as plt

# # Step 1: Read the image using OpenCV
# image = cv2.imread("../../../feature-3dgs/data/fruit/images/IMG_20220408_142309.png")

# # Step 2: Convert from BGR to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

# # # Step 3: Save the image using Matplotlib
# # plt.imsave("save.jpg", image)