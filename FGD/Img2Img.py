"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.abspath('../'))
import StableDiffusion
from StableDiffusion.ldm.util import instantiate_from_config
from StableDiffusion.ldm.models.diffusion.ddim import DDIMSampler
from StableDiffusion.ldm.models.diffusion.plms import PLMSSampler
from StableDiffusion.ldm.models.diffusion.dpm_solver import DPMSolverSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def getMethodParser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    return opt


def stableDiffusionSubpath(path):
    return StableDiffusion.GetStableDiffusionSubpath(path);

def Img2Img(
        experiment,
        ddim_steps=50,
        plms=False,
        dpm_solver=False,
        laion400m=False,
        fixed_code=False,
        ddim_eta=0.0,
        n_iter=1,
        H=512,
        W=512,
        C=4,
        f=8,
        n_samples=1,
        n_rows=0,
        scale=7.5,
        strength=0.6,
        precision="autocast",
        configstr = "configs/stable-diffusion/v1-inference.yaml",
        ckptstr = "models/ldm/stable-diffusion-v1/model.ckpt"
):
    opt = dict(
        experiment=experiment,
        prompt=experiment.guide_text,
        ddim_steps=ddim_steps,
        plms=plms,
        strength=strength,
        dpm_solver=dpm_solver,
        laion400m=laion400m,
        fixed_code=fixed_code,
        ddim_eta=ddim_eta,
        n_iter=n_iter,
        H=H,
        W=W,
        C=C,
        f=f,
        n_samples=n_samples,
        n_rows=n_rows,
        scale=scale,
        precision=precision,
        config = stableDiffusionSubpath(configstr),
        ckpt = stableDiffusionSubpath(ckptstr)
    )

    print('strength', strength)
    print(opt)
    seed_everything(experiment.seed)

    config = OmegaConf.load(f"{opt['config']}")
    model = load_model_from_config(config, f"{opt['ckpt']}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt["plms"]:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

#     os.makedirs(opt.outdir, exist_ok=True)
#     outpath = opt.outdir

    batch_size = n_samples
    n_rows = opt["n_rows"] if opt["n_rows"] > 0 else batch_size

    sampler.make_schedule(ddim_num_steps=opt["ddim_steps"], ddim_eta=opt["ddim_eta"], verbose=False)

    assert 0. <= opt["strength"] <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt["strength"] * opt["ddim_steps"])
    print(f"target t_enc is {t_enc} steps")
    
    ###############
    # Revise by Zeqi
    ###############
    prompt = experiment.guide_text
    assert prompt is not None
    data = [batch_size * [prompt]]
    sampler.opt = opt
    experiment.setGuideLatent(model)
    init_latent = experiment.guide_image.latent_im.view(-1, C, H//f, W//f)
    print('init_latent.size()', init_latent.size())
    ###############

    precision_scope = autocast if opt["precision"] == "autocast" else nullcontext
    start = time.time()
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                
                all_samples = list()
                for n in trange(opt["n_iter"], desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt["scale"] != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        print('z_enc', z_enc.size())
#                         enc_np=z_enc.detach().cpu().numpy()
#                         path='../../stable-diffusion/'+'_'.join([str(opt["strength"]), experiment.guide_text, str(experiment.target_detail),'.npy'])
#                         print(path)
#                         z_enc_np=np.load(path)
#                         z_enc = torch.from_numpy(z_enc_np).to(experiment.device)
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt["scale"],
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.get("skip_save"):
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                experiment.grid_results.append(x_sample.astype(np.uint8));


    experiment.exe_time = time.time()-start

#     print(f"Your samples are ready and waiting for you here: \n"
#           f" \nEnjoy.")


if __name__ == "__main__":
    main()
