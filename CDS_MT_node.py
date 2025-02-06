# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf

from .quick_start.CSD_MT.model import CSD_MT
from .quick_start.faceutils.face_parsing.model import BiSeNet
from .quick_start.CSD_MT_eval import makeup_transfer256

from .node_utils import  tensor2pil_upscale
import folder_paths


MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
CSDMT_weigths_path = os.path.join(folder_paths.models_dir, "CSDMT")
if not os.path.exists(CSDMT_weigths_path):
    os.makedirs(CSDMT_weigths_path)
folder_paths.add_model_folder_path("CSDMT", CSDMT_weigths_path)



class CSDMTLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt": (["none"] + folder_paths.get_filename_list("CSDMT"),),
                "bisenet_ckpt": (["none"] + folder_paths.get_filename_list("CSDMT"),),
            },
        }

    RETURN_TYPES = ("MODEL_CSDMT",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "CSDMT"

    def loader_main(self,ckpt,bisenet_ckpt):

        # load model
        opts_origin = {
                "data_root":'../MT-Dataset/images',
                "lr": 0.0002,
                "lr_policy": "lambda",
                "batch_size": 1,
                "gpu": 0,
                "weight_semantic": 0.,
                "weight_corr": 1.,
                "weight_identity": 0.1, 
                "weight_back": 5.,
                "weight_adv": 2.,
                "weight_contrastive": 1.,
                "weight_self_recL1": 10.,
                "weight_cycleL1": 10.,
                "gan_mode": "original",
                "phase" : "test",
                "dis_scale": 3,
                "dis_norm": "None",
                "dis_sn":True,  
                "G_sn": True,
                "semantic_dim": 10,
                "init_type": "normal",
                "resume": "None",
                "input_dim": 3,
                "output_dim": 3,
                "resize_size": 256,
                "crop_size": 256,
                "flip": True,
                "nThreads":4,
                "name": "CSD-MT",
                "log_dir": "./logs",  
                "model_dir": "./weights", 
                "img_dir": "./images",
                "log_freq": 10,
                "img_save_freq": 1,
                "model_save_freq": 50,
                "n_ep": 500,
                "n_ep_decay": 250,

            }
        opts = OmegaConf.create(opts_origin)
        print("***********Load model ***********")

        # load face_parsing model
        n_classes = 19
        bisnet_ckpt =os.path.join(CSDMT_weigths_path, "resnet18-5c106cde.pth")
        if not os.path.exists(bisnet_ckpt):
            bisnet_ckpt=None
        face_paseing_model = BiSeNet(n_classes,{"path":bisnet_ckpt})
        save_pth = folder_paths.get_full_path("CSDMT",bisenet_ckpt)
        face_paseing_model.load_state_dict(torch.load(save_pth,map_location='cpu'))
        face_paseing_model.eval()

        makeup_model = CSD_MT(opts)
        ep0, total_it = makeup_model.resume(folder_paths.get_full_path("CSDMT",ckpt))
        makeup_model.eval()

        print("***********Load model done ***********")
        gc.collect()
        torch.cuda.empty_cache()
        return ({"makeup_model": makeup_model,"face_paseing_model": face_paseing_model,"opts": opts},)
    
class CSDMTSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CSDMT",),
                "non_makeup_image": ("IMAGE",), #[b,h,w,c]
                "makeup_image": ("IMAGE",), #[b,h,w,c]
                "resize_size": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                }}
                         
        
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "sampler_main"
    CATEGORY = "CSDMT"
    
    def sampler_main(self, model,non_makeup_image,makeup_image,resize_size):
        opts=model.get("opts")
        makeup_model=model.get("makeup_model")
        face_paseing_model=model.get("face_paseing_model")
        opts.resize_size=resize_size
        non_makeup_image=tensor2pil_upscale(non_makeup_image, resize_size, resize_size)
        makeup_image=tensor2pil_upscale(makeup_image, resize_size, resize_size)
        
        image=makeup_transfer256(non_makeup_image, makeup_image,makeup_model,opts,face_paseing_model)
        gc.collect()
        torch.cuda.empty_cache()
        image=torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return (image,)



NODE_CLASS_MAPPINGS = {
    "CSDMTLoader":CSDMTLoader,
    "CSDMTSampler":CSDMTSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSDMTLoader":"CSDMTLoader",
    "CSDMTSampler":"CSDMTSampler",
}
