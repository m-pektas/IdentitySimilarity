import cv2
from detector import FastMtCnnClient
import torch
import numpy as np
from idsim.loss import IdentitySimilarity
import torchvision.transforms as transforms

client = FastMtCnnClient()
IS = IdentitySimilarity()

img1 = cv2.imread("r1.jpg")
img2 = cv2.imread("r2.jpeg")

result1 = client.detect_faces(img1)
result2 = client.detect_faces(img2)

tensorImg1 = transforms.ToTensor()(img1).unsqueeze(0)
tensorImg2 = transforms.ToTensor()(img2).unsqueeze(0)
v1 =  IS.extract_identity(tensorImg1, result1[0].keypoints)
v2 =  IS.extract_identity(tensorImg2, result2[0].keypoints)
sim = IS.forward_v2v(v1, v2)
print("sim_im2im :", sim)