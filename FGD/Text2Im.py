import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

sys.path.insert(0, os.path.abspath('../'))
import StableDiffusion
from StableDiffusion.ldm.util import instantiate_from_config
from StableDiffusion.ldm.models.diffusion.ddim import DDIMSampler
from StableDiffusion.ldm.models.diffusion.plms import PLMSSampler
from StableDiffusion.ldm.models.diffusion.dpm_solver import DPMSolverSampler




# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import acimops

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def load_img(path):
    image = Image.open(path).convert("RGB").resize((512,512))
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept


###############
# Revise by Zeqi
###############
def tensor2img(tensor):
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu()
    tensor = (255 * (tensor + 1)/ 2).round().clamp(0, 255).numpy()
    # np.save('64face_step30.npy', tensor_np)
    tensor_np = tensor.astype('uint8')[0]
    return tensor_np

def img2tensor(img):
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).float()
    img_tensor = img_tensor * 2 -1
    return img_tensor
    ###############


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
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
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
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
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
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
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
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
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
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
#     ###############
#     # Revise by Zeqi
#     ###############
#     parser.add_argument('--target_img', type=str, default=None)
#     parser.add_argument('--t_total', type=int, default=50)
#     parser.add_argument('--t_start', type=int, default=50)
#     parser.add_argument('--t_end', type=int, default=0)
#     parser.add_argument('--spatial_std', type=int, default=5)
#     parser.add_argument('--value_std', type=float, default=0.3)
#     parser.add_argument('--structure', type=float, default=4.5)
#     ###############

    opt = parser.parse_args()
    return opt

def stableDiffusionSubpath(path):
    return StableDiffusion.GetStableDiffusionSubpath(path);

def Text2Im(
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
        # seed=42,
        precision="autocast",
        configstr = "configs/stable-diffusion/v1-inference.yaml",
        ckptstr = "models/ldm/stable-diffusion-v1/model.ckpt"
):
    opt = dict(
        experiment=experiment,
        prompt=experiment.guide_text,
        ddim_steps=ddim_steps,
        plms=plms,
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
        # seed=experiment,
        precision=precision,
        config = stableDiffusionSubpath(configstr),
        ckpt = stableDiffusionSubpath(ckptstr)
    )

    print(opt);
    if laion400m:
        print("Falling back to LAION 400M model...")
        opt["config"] = stableDiffusionSubpath("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        opt["ckpt"] = stableDiffusionSubpath("models/ldm/text2img-large/model.ckpt")
        opt["outdir"] = stableDiffusionSubpath("outputs/txt2img-samples-laion400m")

    # seed_everything(opt.seed)
    seed_everything(experiment.seed)

    config = OmegaConf.load(f"{opt['config']}")
    model = load_model_from_config(config, f"{opt['ckpt']}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if dpm_solver:
        sampler = DPMSolverSampler(model)
        print("Using DPM");
    elif plms:
        sampler = PLMSSampler(model)
        print("Using plms");
    else:
        sampler = DDIMSampler(model)
        print("Using DDIM");

    # os.makedirs(opt.outdir, exist_ok=True)
    # outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = opt["n_rows"] if opt["n_rows"] > 0 else batch_size



    ###############
    # Revise by Zeqi
    ###############
    prompt = experiment.guide_text
    assert prompt is not None
    data = [batch_size * [prompt]]
    if opt["fixed_code"]:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)
    else: 
        if isinstance(experiment.xT_latent, str):#needs inversion
            ddim_inversion_steps = 999
            condition_inversion = model.get_learned_conditioning([""])
            seed_everything(-1)
            init_image = load_img(experiment.guide_image.absolute_file_path).to(experiment.device)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            experiment.guide_image.latent_dim = init_latent.size();
            [_, ch, h, w] = experiment.guide_image.latent_dim;
            experiment.guide_image.latent_im = init_latent.reshape((ch, 1, h*w))
            filename = experiment.xT_latent
                
            experiment.xT_latent, _ = sampler.encode_ddim(init_latent, ddim_inversion_steps,\
                                                condition_inversion, unconditional_conditioning=condition_inversion,\
                                                unconditional_guidance_scale=1.0, invert_portion=experiment.invert_portion)
            torch.save(experiment.xT_latent.detach().cpu(), filename)
            print('inversion saved', filename)
        start_code = experiment.xT_latent    
            
    sampler.opt = opt
    experiment.setGuideLatent(model)
    t_enc = int(experiment.invert_portion * ddim_steps)
    if not plms:
        sampler.make_schedule(ddim_num_steps=opt["ddim_steps"], ddim_eta=opt["ddim_eta"], verbose=False)
#     if experiment.guide_image:
#         # target = img2tensor(Image.open(opt.target_img).convert('RGB').resize((64, 64)))
#         target = img2tensor(experiment.guide_image.scaled.fpixels);

#         sampler.target_np = target[0].permute(1, 2, 0).cpu().detach().numpy()
#         sampler.sigmas = [experiment.sigmas[0], experiment.sigmas[1], experiment.sigmas[2]];

#         if(experiment.guide_image.latent_im is None):
#             init_image = load_img(experiment.guide_image.absolute_file_path).to(device)
#             #convert the guidance image to latent space
#             init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
#             # _, ch, h, w = init_latent.size()
#             experiment.guide_image.latent_dim = init_latent.size();
#             [_, ch, h, w] = experiment.guide_image.latent_dim;
#             init_latent = init_latent.reshape((ch, 1, h*w))
#             experiment.guide_image.latent_im = init_latent;
#             #1x4x64x64
#             print('guidance image latent feature', init_latent.size())
#         else:
#             [_, ch, h, w] = experiment.guide_image.latent_dim;
#             init_latent = experiment.guide_image.latent_im;

#         # _, ch, h, w = init_latent.size()
#         #cross bilat kernel, size 4x4096

#         # acimops.bilateral.getCrossBilateralMatrix(sampler.target_np.astype('float'), \
#         #                                             sampler.sigmas)
#         if(experiment.guide_image.jbmsigmas == experiment.sigmas):
#             print("Reusing JBM for sigmas {}".format(experiment.sigmas));
#         else:
#             cbm = experiment.guide_image.getCrossBilateralMatrix(experiment.sigmas);
#             experiment.guide_image.setJBM(torch.Tensor(cbm).unsqueeze(0).repeat((ch, 1,1)).to(device), experiment.sigmas);
#             # sampler.target_structure = torch.bmm(init_latent, sampler.cbm).reshape((ch, h, w))

#         if(experiment.guide_image.latent_structure is None):
#             init_latent = init_latent.reshape((ch, 1, h*w))
#             experiment.guide_image.latent_structure = torch.bmm(init_latent, experiment.guide_image.jbm).reshape((ch, h, w))
    ##############

    precision_scope = autocast if precision=="autocast" else nullcontext
    start = time.time()
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt["scale"] != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        if plms:
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c,
                                                             batch_size=n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc,
                                                             eta=ddim_eta,
                                                             x_T=start_code)
                        else:
                            samples_ddim = sampler.decode(start_code, c, t_enc, unconditional_guidance_scale=opt["scale"],
                                                 unconditional_conditioning=uc,)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        has_nsfw_concept = False

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.get("skip_save"):
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                experiment.grid_results.append(x_sample.astype(np.uint8));
                                # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                # base_count += 1

                        # if not opt.skip_grid:
                        #     all_samples.append(x_checked_image_torch)

                # if not opt.skip_grid:
                #     # additionally, save as grid
                #     grid = torch.stack(all_samples, 0)
                #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                #     grid = make_grid(grid, nrow=n_rows)
                #
                #     # to image
                #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #     img = Image.fromarray(grid.astype(np.uint8))
                #     img = put_watermark(img, wm_encoder)
                #     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                #     grid_count += 1

    experiment.exe_time = time.time()-start

    # print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
    #       f" \nEnjoy.")