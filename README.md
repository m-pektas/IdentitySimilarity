# Identity Similarity

This repository can helps researchers that want to use face recognition in their researches. You can easly implement current(August 2022) sota face recognition in your project. I motivated for this repository from 
[LPIPS](https://github.com/richzhang/PerceptualSimilarity).

Models borrowed from [Insigtface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).

**Warning :** Please, be careful when chosing your criterion. Lower is more similar in MSE while higher is more similar in CosineSimilarity.

## Params

- **model**
    - **name**   : model name. it takes r50 and r100 values. Now, r100 supported only.
    - **device** : target device. it takes cuda and cpu variables.
- **ref_points_path** : aligned 5 landmark template. It takes ref. points numpy file path or None.
- **criterion** : Similarity metric. Now, only supported MSE.

## Example

```
import torch
from idsim.loss import IdentitySimilarity

if __name__ == "__main__":

    cfg = {"model": {
        "name": "r100",
        "device": "cuda",
    },
        "ref_points_path": None,
        "criterion": "MSE"
    }
 
    src = np.array([[35.066223, 34.23266],
                    [84.1586, 33.96113],
                    [59.768444, 62.152763],
                    [39.60066, 90.89288],
                    [80.255, 90.66802]], dtype=np.float32)

    IS = IdentitySimilarity(cfg)
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
```
```
# Output
sim_v2v : tensor(0.1647, device='cuda:0') 
sim_im2im : tensor(0.0332, device='cuda:0', grad_fn=<MseLossBackward0>) 
sim_v2im : tensor(0.4590, device='cuda:0', grad_fn=<MseLossBackward0>)
identity vector : tensor([[ 0.1474, -0.6962,  0.2244,  ...,  0.5385,  0.2144,  0.4057],
        [ 0.2828, -0.7189, -0.1680,  ...,  0.7966,  0.2238,  0.4943],
        [ 0.1210, -1.0216, -0.0179,  ...,  0.8538,  0.2481,  0.5686],
        [ 0.1712, -0.5780, -0.0307,  ...,  0.7353,  0.2200,  0.3596],
        [-0.0011, -0.5154,  0.3020,  ...,  0.7727,  0.1472,  0.3377]],
       device='cuda:0', grad_fn=<NativeBatchNormBackward0>)
```

## TODOs

- [] Extract identities with different referance points for each image in batch.
- [] Support different referance points for calculating similarity between two diffirent image batch.
- [] Support r50 version of arcface.
- [] Add cosine similarity as a similarity metric.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request