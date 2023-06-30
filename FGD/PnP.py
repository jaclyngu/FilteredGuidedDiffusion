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
from einops import rearrange
import json
# from pnp_utils import check_safety

sys.path.insert(0, os.path.abspath('../'))
import StableDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# from StableDiffusion.ldm.models.diffusion.plms import PLMSSampler
# from StableDiffusion.ldm.models.diffusion.dpm_solver import DPMSolverSampler

# sys.path.insert(0, os.path.abspath('../../'))
# import importlib  
# foobar = importlib.import_module("plug-and-play.ldm.models.diffusion.ddim")
# # from foobar import DDIMSampler
# DDIMSampler = foobar.DDIMSampler


# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import acimops
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


def getMethodParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '../../plug-and-play/configs/pnp/pnp-real.yaml')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--model_config', type=str, default='configs/stable-diffusion/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--check-safety", action='store_true')
    opt = parser.parse_args()
    return opt

def stableDiffusionSubpath(path):
    return StableDiffusion.GetStableDiffusionSubpath(path);

def PnP(
        experiment,
        ddim_steps=50,
#         plms=False,
#         dpm_solver=False,
#         laion400m=False,
        fixed_code=False,
        ddim_eta=0.0,
#         n_iter=1,
        H=512,
        W=512,
        C=4,
        f=8,
#         n_samples=1,
#         n_rows=0,
#         scale=7.5,
        # seed=42,
        precision="autocast",
        configstr = "configs/stable-diffusion/v1-inference.yaml",
        ckptstr = "models/ldm/stable-diffusion-v1/model.ckpt",
        config='../../plug-and-play/configs/pnp/pnp-real.yaml'
):
    opt = dict(
        experiment=experiment,
        prompt=experiment.guide_text,
        ddim_steps=ddim_steps,
#         plms=plms,
#         dpm_solver=dpm_solver,
#         laion400m=laion400m,
        fixed_code=fixed_code,
        ddim_eta=ddim_eta,
#         n_iter=n_iter,
        H=H,
        W=W,
        C=C,
        f=f,
#         n_samples=n_samples,
#         n_rows=n_rows,
#         scale=scale,
        # seed=experiment,
        precision=precision,
        model_config = stableDiffusionSubpath(configstr),
        ckpt = stableDiffusionSubpath(ckptstr),
        config=config
    )

    exp_config = OmegaConf.load(opt["config"])

    exp_path_root_config = OmegaConf.load("setup.yaml")
    exp_path_root = exp_path_root_config.config.exp_path_root
    
    # read seed from args.json of source experiment
    with open(os.path.join(exp_path_root, exp_config.source_experiment_name, "args.json"), "r") as f:
        args = json.load(f)
        seed = args["seed"]
        source_prompt = args["prompt"]
    negative_prompt = source_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

    seed_everything(seed)
    possible_ddim_steps = args["save_feature_timesteps"]
    assert exp_config.num_ddim_sampling_steps in possible_ddim_steps or exp_config.num_ddim_sampling_steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {exp_config.num_ddim_sampling_steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {exp_config.num_ddim_sampling_steps}"
    ddim_steps = exp_config.num_ddim_sampling_steps if exp_config.num_ddim_sampling_steps is not None else possible_ddim_steps[-1]
    
    model_config = OmegaConf.load(f"{opt['model_config']}")
    model = load_model_from_config(model_config, f"{opt['ckpt']}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt["ddim_eta"], verbose=False) 

    seed = torch.initial_seed()
    opt["seed"] = seed

    translation_folders = [p.replace(' ', '_') for p in exp_config.prompts]
    outpaths = [os.path.join(f"{exp_path_root}/{exp_config.source_experiment_name}/translations", f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
    out_label = f"INJECTION_T_{exp_config.feature_injection_threshold}_STEPS_{ddim_steps}"
    out_label += f"_NP-ALPHA_{exp_config.negative_prompt_alpha}_SCHEDULE_{exp_config.negative_prompt_schedule}_NP_{negative_prompt.replace(' ', '_')}"

    predicted_samples_paths = [os.path.join(outpath, f"predicted_samples_{out_label}") for outpath in outpaths]
    for i in range(len(outpaths)):
        os.makedirs(outpaths[i], exist_ok=True)
        os.makedirs(predicted_samples_paths[i], exist_ok=True)
        # save args in experiment dir
        with open(os.path.join(outpaths[i], "args.json"), "w") as f:
            json.dump(OmegaConf.to_container(exp_config), f)

    def save_sampled_img(x, i, save_paths):
        for im in range(x.shape[0]):
            x_samples_ddim = model.decode_first_stage(x[im].unsqueeze(0))
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
            x_sample = x_image_torch[0]

            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(save_paths[im], f"{i}.png"))

    def ddim_sampler_callback(pred_x0, xt, i):
        save_sampled_img(pred_x0, i, predicted_samples_paths)

    def load_target_features():
        self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
        out_layers_output_block_indices = [4]
        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [exp_config.feature_injection_threshold]
        target_features = []

        source_experiment_out_layers_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")
        source_experiment_qkv_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")
        
        time_range = np.flip(sampler.ddim_timesteps)
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        
        for i, t in enumerate(iterator):
            current_features = {}
            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                if i <= int(output_block_self_attn_map_injection_threshold):
                    output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                    output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                if i <= int(feature_injection_threshold):
                    output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_out_layers'] = output

            target_features.append(current_features)

        return target_features

    batch_size = len(exp_config.prompts)
    prompts = exp_config.prompts
    assert prompts is not None

    ###############
    # Revise by Zeqi
    ###############
    if opt["fixed_code"]:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)
    else:
        start_code = experiment.xT_latent
    sampler.opt = opt
    experiment.setGuideLatent(model)
    ###############
    
#     start_code_path = f"{exp_path_root}/{exp_config.source_experiment_name}/z_enc.pt"
#     start_code = torch.load(start_code_path).cuda() if os.path.exists(start_code_path) else None
#     if start_code is not None:
#         start_code = start_code.repeat(batch_size, 1, 1, 1)

    precision_scope = autocast if opt["precision"]=="autocast" else nullcontext
    injected_features = load_target_features()
    unconditional_prompt = ""
    start=time.time()
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                nc = None
                if exp_config.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                    nc = model.get_learned_conditioning(batch_size * [negative_prompt])
                if not isinstance(prompts, list):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [opt["C"], opt["H"] // opt["f"], opt["W"] // opt["f"]]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 negative_conditioning=nc,
                                                 batch_size=len(prompts),
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=exp_config.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt["ddim_eta"],
                                                 x_T=start_code,
                                                 img_callback=ddim_sampler_callback,
                                                 injected_features=injected_features,
                                                 negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                 negative_prompt_schedule=exp_config.negative_prompt_schedule,
                                                 )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
#                 if opt.check_safety:
#                     x_samples_ddim = check_safety(x_samples_ddim)
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                sample_idx = 0
                for k, x_sample in enumerate(x_image_torch):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    experiment.grid_results.append(img);
#                     img.save(os.path.join(outpaths[k], f"{out_label}_sample_{sample_idx}.png"))
                    sample_idx += 1

    print(f"PnP results saved in: {'; '.join(outpaths)}")
    experiment.exe_time = time.time()-start

if __name__ == "__main__":
    start=time.time()
    main()
    print(time.time()-start)
    
