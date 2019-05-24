#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:30:11 2019

@author: clytie
"""

if __name__ == "__main__":
    from dcgan import DCGAN
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    
    
    image_path = os.listdir("faces")
    datas = []
    for path in tqdm(image_path):
        if "jpg" in path or "png" in path:
            datas.append(cv2.imread(f"faces/{path}"))
    datas = np.asarray(datas)
    
    img_dim = (96, 96, 3)
    dcgan = DCGAN(img_dim)
    dcgan.train(datas)

