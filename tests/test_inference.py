import torch
import numpy as np
from idsim import IdentitySimilarity

class TestInference:

    idsim = IdentitySimilarity(model_name="r50", device="cpu", criterion="MSE")
    vector_a = torch.rand(1, 512)
    vector_b = torch.rand(1, 512)
    image_a = torch.rand(5, 3, 128, 128)
    image_b = torch.rand(5, 3, 128, 128)

    def test_same_vectors(cls):
        sim_v2v = cls.idsim.forward_v2v(cls.vector_a, cls.vector_a)
        assert sim_v2v.item() == 0, "Similarity between two same vectors must be 0"
    
    def test_different_vectors(cls):
        sim_v2v = cls.idsim.forward_v2v(cls.vector_a, cls.vector_b)
        assert sim_v2v.item() != 0, "Similarity between two different vectors cannot be be 0"

    def test_same_images(cls):
        src = np.array([[35.066223, 34.23266],
                  [84.1586, 33.96113],
                  [59.768444, 62.152763],
                  [39.60066, 90.89288],
                  [80.255, 90.66802]], dtype=np.float32)
        cls.idsim.set_ref_point(src)
        sim_v2img = cls.idsim.forward_img2img(cls.image_a, cls.image_a)
        assert sim_v2img.item() == 0, "Similarity between two same images must be 0"

    def test_different_images(cls):
        src = np.array([[35.066223, 34.23266],
                  [84.1586, 33.96113],
                  [59.768444, 62.152763],
                  [39.60066, 90.89288],
                  [80.255, 90.66802]], dtype=np.float32)
        cls.idsim.set_ref_point(src)
        sim_v2img = cls.idsim.forward_img2img(cls.image_a, cls.image_b)
        assert sim_v2img.item() != 0, "Similarity between two different images cannot be be 0"
