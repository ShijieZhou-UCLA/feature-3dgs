import torch
import torch.nn as nn
import clip
from PIL import Image
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms


class CLIPEditor(object):
    def __init__(self):
        super(CLIPEditor, self).__init__()
        self.device = "cuda"
        self.model, _preprocess = clip.load("ViT-B/32", device=self.device, download_root="/tmp/tmp_clip")
        self.model = self.model.float()
        self.text_features = None
        self.text_filter_features = None

    def preprocess(self, image, stochastic=0):
        # image: nchw, range [0, 1]
        if stochastic:
            images = []
            for i in range(stochastic):
                _image = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomRotation(20, interpolation=Image.BILINEAR),
                    transforms.GaussianBlur(3, sigma=(0.01, 2.0)),
                    transforms.Resize(self.model.visual.input_resolution, interpolation=Image.BICUBIC),
                ])(image)
                images.append(_image)
            image = torch.cat(images, dim=0)
            """
            print(image.shape)
            for img in images:
                import numpy as np
                import time
                import imageio
                rgb_pred = (img.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)  # (h,w,c)
                imageio.imsave('./aug_tmpdebug_____{}.png'.format(time.time()), rgb_pred)
            """
        else:
            image = transforms.Resize(self.model.visual.input_resolution, interpolation=Image.BICUBIC)(image)
        image = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(image)
        return image

    def encode_image(self, image, preprocess=True, stochastic=0):
        if preprocess:
            image = self.preprocess(image, stochastic=stochastic)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text_list):
        with torch.no_grad():
            texts = clip.tokenize(text_list).to(self.device)
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
