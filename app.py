import cv2
from idsim import IdentitySimilarity
from tqdm import tqdm

idsim = IdentitySimilarity(model_name="r50", criterion="Cosine")
img1 = cv2.imread("b.jpg")
img2 = cv2.imread("a.jpg")
v1 = idsim.extract_identity(img1) 
v2 = idsim.extract_identity(img2)
sim = idsim.forward_v2v(v1,v2)
print("Similarity :", sim)