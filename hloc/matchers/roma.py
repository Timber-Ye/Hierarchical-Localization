import warnings

import torch
from romatch.utils.utils import tensor_to_pil
from romatch import roma_indoor, roma_outdoor

from ..utils.base_model import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoMa(BaseModel):
    default_conf = {
        "max_num_matches": None,
    }

    def _init(self, conf):
        # Create model
        self.net = roma_indoor(device=device, coarse_res=560, upsample_res=(864, 1152))

    def _forward(self, data):
        im1 = data["image0"]
        im2 = data["image1"]

        H_A, W_A = im1.shape[-2:]
        H_B, W_B = im2.shape[-2:]

        warp, certainty = self.net.match(im1, im2, device=device)
        matches, certainty = self.net.sample(warp, certainty)
        kpts0, kpts1 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        top_k = self.conf["max_num_matches"]
        if top_k is not None and len(certainty) > top_k:
            keep = torch.argsort(certainty, descending=True)[:top_k]
            kpts0, kpts1 = kpts0[keep], kpts1[keep]
            certainty = certainty[keep]

        pred = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "scores": certainty,
        }

        return pred
