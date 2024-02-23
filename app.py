import torch
import numpy as np
from idsim.loss import IdentitySimilarity

if __name__ == "__main__":

 
    src = np.array([[35.066223, 34.23266],
                    [84.1586, 33.96113],
                    [59.768444, 62.152763],
                    [39.60066, 90.89288],
                    [80.255, 90.66802]], dtype=np.float32)

    IS = IdentitySimilarity()
    IS.set_ref_point(src)

    # dummy variables
    v1 = torch.rand(1, 512)
    v2 = torch.rand(1, 512)
    im1 = torch.rand(5, 3, 128, 128)
    im2 = torch.rand(5, 3, 128, 128)

    # useful functions
    sim_v2v = IS.forward_v2v(v1, v2)
    sim_im2im = IS.forward_img2img(im1, im2)
    sim_v2im = IS.forward_v2img(v1, im1)
    print("\nsim_v2v :", sim_v2v,
          "\nsim_im2im :", sim_im2im,
          "\nsim_v2im :", sim_v2im)

    identity = IS.extract_identity(im2, src)
    print("identity vector :", identity)