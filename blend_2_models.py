import os
import pickle
from copy import deepcopy
from itertools import product

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def to_numpy(t: torch.Tensor):
    """Convert model output from Torch Tensor to numpy array."""

    assert t.ndim == 4

    # Inverse normalize [-1, 1] -> [0, 1], rescale [0, 1] -> [0, 255]
    out = (torch.clamp(t * 0.5 + 0.5, 0, 1) * 255) \
        .permute(0, 2, 3, 1) \
        .detach() \
        .cpu().numpy() \
        .astype(np.uint8)
    return out


def vflip_spatial_weights(state_dict):
    """
    Make a copy of the model weights with any 3 or 4 dimensional weightz vertical flipped.
    This was needed because a model trained using StyleGAN2-ada was producing vertically flipped images.
    """

    new_state_dict = deepcopy(state_dict)
    for k, w in new_state_dict.items():
        if w.ndim in [3, 4]:
            new_state_dict[k] = torch.flip(w, dims=[-2])
    return new_state_dict


def load_model(model_pkl_path, device=None):
    """Load StyleGAN2 model from pickle checkpoint file."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(model_pkl_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
    return G


def generate(G, n=1):
    """Generate a batch of images using StyleGAN2 generator model."""

    z = torch.randn([n, G.z_dim]).to(device)    # latent codes
    # class labels (not used in this example)
    c = None

    img = G(z, c)
    img = np.squeeze(to_numpy(img))

    return img


def generate_n_and_save_to_dir(G, outdir, n=5, prefix=""):
    """Generate some amount of images using StyleGAN2 generator model and save them to an output directory."""

    os.makedirs(outdir, exist_ok=True)

    imgs = generate(G, n)
    for img in imgs:
        Image.fromarray(img).save(os.path.join(outdir, f"{prefix}{i}.jpg"))


def get_conv_weights_names(G):
    """Get convolutional layer weights names from StyleGAN2 generator model."""

    state_dict = G.state_dict()

    keys = state_dict.keys()
    keys = [k for k in keys if "conv" in k]
    keys = [k for k in keys if k.endswith(".weight") or k.endswith(".bias")]
    keys = [k for k in keys if "affine" not in k]
    return keys


def blend_2_models(G1, G2,
                   blocks=[
                       "b4",
                       "b8",
                       "b16",
                       "b32",
                       "b64",
                       "b128",
                   ],
                   weights=[
                       0.5,
                       0.5,
                       0.5,
                       0.5,
                       0.5,
                       0.5
                   ],
                   # XXX What to blend? Weights only or both weights and bias?
                   weight_types=["weight", "bias"]
                   ):
    """Blend two models by blending their convolutional layer weights."""

    assert len(blocks) == len(weights)

    state_dict_1 = G1.state_dict()
    state_dict_2 = G2.state_dict()
    state_dict_blend = deepcopy(state_dict_1)

    for block, alpha in zip(blocks, weights):
        for i in range(2):
            for wt in weight_types:
                # Get weights name
                w_name = f"synthesis.{block}.conv{i}.{wt}"

                # If weights name is valid, blend weights from two different model
                if w_name in state_dict_1 and w_name in state_dict_2:
                    w_model_1 = state_dict_1[w_name]
                    w_model_2 = state_dict_2[w_name]

                    w_blend = alpha * w_model_1 + (1 - alpha) * w_model_2

                    print(f"Blending weights for {w_name}")
                    state_dict_blend[w_name] = w_blend

    return state_dict_blend


if __name__ == "__main__":

    G_blend = load_model('./celebahq_128_snapshot_004435.pkl')

    # Load model #1
    G_celebahq = load_model('./celebahq_128_snapshot_004435.pkl')

    # Load model #2
    G_punk = load_model(
        'punk_rgb_128_augment_factor_5/00000-punk_rgb_128_augment_factor_5-auto1/network-snapshot-003400.pkl')
    G_punk.load_state_dict(torch.load("./punk_rgb_128_vflip.pth",
                                      map_location="cpu"))

    # Generate images using various combination of weights
    out_dir = "test"
    r = np.linspace(0, 1, 6)
    for weights in tqdm(product(r, r, r, r, r, r)):
        conf_name = ",".join(map(lambda v: f"{v:.1f}", list(weights)))

        # Blend weights of two models
        blended_weights = blend_2_models(G_celebahq, G_punk, weights=weights)
        G_blend.load_state_dict(blended_weights)

        generate_n_and_save_to_dir(G_blend, 5
                                   out_dir,
                                   prefix=conf_name + "-")

    print("Done!")
