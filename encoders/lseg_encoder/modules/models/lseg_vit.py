import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import clip
import os
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch/"
os.environ["TORCH_HOME"] = "/tmp/torch/"

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_vit(pretrained, x, preresize=None):
    b, c, h, w = x.shape
    
    # encoder
    # print(x.shape, "x.shape")
    # torch.Size([1, 3, 480, 480]) x.shape
    if preresize is not None:
        # preresize = [420, 420]  # 480/16*14
        # _x = torch.nn.functional.interpolate(x, size=preresize, mode="bilinear", align_corners=False)
        _x = torch.nn.functional.interpolate(x, size=[int(h * preresize), int(w * preresize)], mode="bilinear", align_corners=False)
        glob = pretrained.model.forward_flex(_x)
        #print(_x.shape, "x.shape resize")
        #torch.Size([1, 3, 336, 336]) x.shape resize
        # glob = pretrained.model.forward_flex(x)
    else:
        glob = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    # print(x.shape)# torch.Size([1, 3, 480, 480])

    #print(layer_1.shape)
    #print(layer_2.shape)
    #print(layer_3.shape)
    #print(layer_4.shape)

    if layer_1.shape[0] != b:
        # maybe clip is used. LNC -> NLC
        layer_1 = layer_1.permute(1, 0, 2)
        layer_2 = layer_2.permute(1, 0, 2)
        layer_3 = layer_3.permute(1, 0, 2)
        layer_4 = layer_4.permute(1, 0, 2)

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    #print(layer_1.shape)
    #print(layer_2.shape)
    #print(layer_3.shape)
    #print(layer_4.shape)
    # base
    #torch.Size([1, 901, 1024])
    #torch.Size([1, 901, 1024])
    #torch.Size([1, 901, 1024])
    #torch.Size([1, 901, 1024])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])

    # clip
    #torch.Size([577, 1, 1024])
    #torch.Size([577, 1, 1024])
    #torch.Size([577, 1, 1024])
    #torch.Size([577, 1, 1024])
    #torch.Size([1, 1024, 576])
    #torch.Size([1, 1024, 576])
    #torch.Size([1, 1024, 576])
    #torch.Size([1, 1024, 576])
    # clip 420resize
    #torch.Size([901, 1, 1024])
    #torch.Size([901, 1, 1024])
    #torch.Size([901, 1, 1024])
    #torch.Size([901, 1, 1024])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    #torch.Size([1, 1024, 900])
    
    unflatten = nn.Sequential(
        # nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0],]),)
        nn.Unflatten(2, torch.Size([h // 16, w // 16,]),)
    )
    #print(layer_1.shape)
    #print(h, w, pretrained.model.patch_size)
    #print([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]])

    # baseline
    #torch.Size([1, 1024, 900])
    #480 480 [16, 16]
    #[30, 30]

    # clip 420
    #torch.Size([1, 1024, 900])
    #480 480 [14, 14]
    #[34, 34]

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))
    # print(posemb.shape, gs_h, gs_w, gs_old, self.start_index, posemb_tok.shape, posemb_grid.shape)
    #torch.Size([1, 577, 1024]) 30 30 24 1 torch.Size([1, 1, 1024]) torch.Size([576, 1024])
    #torch.Size([1, 257, 1024]) 16 16 16 1 torch.Size([1, 1, 1024]) torch.Size([256, 1024])
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex_clip(self, x):
    b, c, h, w = x.shape
    # print(self)
    #if hasattr(self, "pos_embed"):
    #    pos_embed = self._resize_pos_embed(
    #        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    #    )
    #else:
    # CLIP applies first applies convlution before pos emb and patching. it reduces the width and height. (336->224)
    #print(self.input_resolution, self.positional_embedding.shape, self.patch_size, h, w)
    #224 torch.Size([257, 1024]) [14, 14] 336 336
    pos_embed = self._resize_pos_embed(
        self.positional_embedding[None], h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]
    # print(pos_embed.shape, self.positional_embedding.shape) # torch.Size([1, 577, 1024]) torch.Size([257, 1024])
    #print(x.shape)
    #torch.Size([1, 257, 1024]) torch.Size([257, 1024])
    #torch.Size([1, 3, 336, 336])

    #torch.Size([1, 257, 1024]) 16 16 16 1 torch.Size([1, 1, 1024]) torch.Size([256, 1024])
    #torch.Size([1, 257, 1024]) torch.Size([257, 1024])
    #torch.Size([1, 3, 336, 336])
    #torch.Size([1, 1024, 24, 24])
    #torch.Size([1, 1024, 576])
    #torch.Size([1, 576, 1024])
    #torch.Size([1, 577, 1024]) after cat
    #torch.Size([1, 577, 1024])
    #torch.Size([1, 577, 1024]) after lnpre
    #torch.Size([577, 1, 1024])
    #torch.Size([577, 1, 1024]) aftertrans
    #torch.Size([1, 577, 1024])
    #torch.Size([1, 1024])
    #torch.Size([1, 768]) out
    # print(x.shape) # [1, 3, 336, 336]
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    # print(x.shape) # [1, 1024, 24, 24]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    # print(x.shape) # [1, 1024, 576]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    # print(x.shape) # [1, 576, 1024]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    # print(x.shape, "after cat") # [1, 577, 1024]
    # x = x + self.positional_embedding.to(x.dtype)
    x = x + pos_embed
    # print(x.shape)
    x = self.ln_pre(x)
    # print(x.shape, "after lnpre")
    x = x.permute(1, 0, 2)  # NLD -> LND
    # print(x.shape)
    x = self.transformer(x)
    # print(x.shape, "aftertrans")
    x = x.permute(1, 0, 2)  # LND -> NLD
    # print(x.shape)
    x = self.ln_post(x[:, 0, :])
    # print(x.shape)
    if self.proj is not None:
        x = x @ self.proj
    # print(x.shape, "out")
    return x
"""

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x
"""

def forward_flex(self, x):
    b, c, h, w = x.shape
    if not hasattr(self, "pos_embed"):
        return forward_flex_clip(self, x)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]
    
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

def _make_fullclip_vitl14_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    # clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root="/tmp/")
    clip_pretrained, _ = clip.load("ViT-L/14", device='cuda', jit=False, download_root="/tmp/")
    clip_pretrained = clip_pretrained.float()
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks == None else hooks
    # pretrained = _make_vit_b16_backbone(
    pretrained = _make_clipvisual_vit_l14_backbone(
        # model,
        clip_pretrained.visual,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained
    # TODO: resize at input (384->336) 384/16 = 336/14 = 20
    # return clip_pretrained

def _make_pretrained_clip_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root="/tmp/")
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained


def _make_pretrained_clipRN50x16_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    clip_pretrained, _ = clip.load("RN50x16", device='cuda', jit=False, download_root="/tmp/")
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained


def _make_pretrained_clip_vitb32_384(pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False):
    clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root="/tmp/")
    model = timm.create_model("vit_base_patch32_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    
    pretrained = _make_vit_b32_backbone(
        model, 
        features=[96, 192, 384, 768], 
        hooks=hooks, 
        use_readout=use_readout,
        enable_attention_hooks=False,
    )
    return clip_pretrained, pretrained


def _make_vit_b32_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()
    
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    pretrained.model.patch_size = [32, 32]
    pretrained.model.start_index = start_index

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=8,
            stride=8,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[2],
            out_channels=features[2],
            kernel_size=2,
            stride=2,
            padding=0,
            # output_padding=output_padding,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )
    
    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


# def _make_vit_b16_backbone(
def _make_clipvisual_vit_l14_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model
    """
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    """
    i = 1
    print("_make_clipvisual_vit_l14_backbone")
    for l, block in enumerate(pretrained.model.transformer.resblocks):
        if l in hooks:
            block.register_forward_hook(get_activation(str(i)))
            i += 1
            print(i, str(block)[:50] + "...")
    assert i == len(hooks) + 1, (i, hooks)

    pretrained.activations = activations

    assert not enable_attention_hooks
    """
    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention
    """

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    # pretrained.model.patch_size = [16, 16]
    pretrained.model.patch_size = [14, 14]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained
