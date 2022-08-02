from genericpath import exists
import torch
import numpy as np
from kornia.geometry.transform import warp_affine
from skimage import transform as trans 
from .core import get_model
from .utils.parser_util import dictToMunch
import os
import gdown


class IdentitySimilarity:
    
    def __init__(self, cfg):

        if isinstance(cfg, dict):
            cfg = dictToMunch(cfg)

        self.cfg                            = cfg
        self.device                         = self.cfg.model.device
        self.root = "/".join(os.path.abspath(__file__).split("/")[:-1])

        self.check_models()
        self.criterion                      = self.__init_criterion()
        self.arcface                        = self.__init_arcface()
        self.src, self.dst, self.dest_size  = self.__init_src_dst_points()
        self.__init_translation_matrix()


    
    def check_models(self):
        self.model_paths = {
            "r50" : [f"{self.root}/models/r50.pth","-"],
            "r100": [f"{self.root}/models/glint360k_cosface_r100_fp16_01.pth",
                     "https://drive.google.com/u/0/uc?id=13_xDly_05M0rBkoikaiaBJpaIh9NO4q6"]
        }
        
        for k, p in self.model_paths.items():
            if os.path.exists(p[0]):
                print(f"{k} : ok!")
            else:
                
                if k=="r50": continue
                
                print(f"{k} : downloading ...")
                gdown.download(p[1],p[0])
        

    def __init_criterion(self):
        
        if self.cfg.criterion is not  None:

            if self.cfg.criterion == "MSE":
                criterion = torch.nn.MSELoss()
            else:
                raise Exception("Unsupported criterion metric !!!!")
        else:
            raise Exception("Criterion metric cannot be None !!!!")


        return criterion
        
    def __init_arcface(self):

        name = self.cfg.model.name
        path = self.model_paths[name]

        net = get_model(name, fp16=True)
        net.load_state_dict(torch.load(path[0]))
        net.eval()
        net.to(self.device)
        return net

    
    def __init_src_dst_points(self):
        dst = np.array([[30.2946,51.6963],
                    [65.5318,51.5014],
                    [48.0252,71.7366],
                    [33.5493,92.3655],
                    [62.7299,92.2041]], dtype=np.float32)

        if self.cfg.ref_points_path is not None:
            src = np.load(self.cfg.ref_points_path)
        else:
            src = None

        return src, dst, 112

    def __init_translation_matrix(self):
        M = None
        if self.src is not None:
            similarity_transform = trans.SimilarityTransform()
            similarity_transform.estimate(self.src, self.dst)
            M = torch.tensor(similarity_transform.params[0:2, :])
            M = M.to(self.device )
            self.M = M

    def set_ref_point(self, ref_point):
        self.src = ref_point
        self.__init_translation_matrix()

    def get_translation_matrix(self, ref_points):
        similarity_transform = trans.SimilarityTransform()
        similarity_transform.estimate(ref_points, self.dst)
        M = torch.tensor(similarity_transform.params[0:2, :])
        M = M.to(self.device )
        return M
    

    def extract_identity(self, im, ref_points):
        M = self.get_translation_matrix(ref_points)
        M = M.unsqueeze(0).repeat(im.size(0),1,1).float()
        im_align = warp_affine(im.to(self.device).float(), 
                               M,
                               dsize=(112,112))
        im_id = self.arcface(im_align)
        return im_id

    def forward_v2v(self, id1, id2):
        return self.criterion(id1.to(self.device), id2.to(self.device))
    
    def forward_v2img(self, id1, im):
        M = self.M.unsqueeze(0).repeat(im.size(0),1,1).float()
        id1 = id1.repeat(im.size(0),1,).to(self.device).float()
        im_align = warp_affine(im.to(self.device).float(), 
                               M,
                               dsize=(112,112))
        im_id = self.arcface(im_align)
        return self.criterion(id1, im_id)
    
    def forward_img2img(self, im1, im2):
        
        M = self.M.unsqueeze(0).repeat(im1.size(0),1,1).float()
        
        im1_align = warp_affine(im1.to(self.device).float(), M, dsize=(112,112))
        im2_align = warp_affine(im2.to(self.device).float(), M, dsize=(112,112))
        im1_id = self.arcface(im1_align)
        im2_id = self.arcface(im2_align)        
        return self.criterion(im1_id, im2_id)
    
    
    
        
        