from PIL import Image
import torch as th
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
from scipy import ndimage
import random
import cv2

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


def show_images(args, batch: th.Tensor, suffix='samples'):
    """ Display a batch of images inline. """
    out_np = tensor2img(batch)

    # scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # shape_str = "x".join([str(x) for x in reshaped.shape])
    # import pdb; pdb.set_trace()
    out_path = args.direction + "_results/" + suffix+'.png'
    Image.fromarray(out_np).save(out_path)
    # print(f"saving to {out_path}")
    # np.savez(out_path, reshaped)


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


def main(args, model, diffusion, model_up, diffusion_up):
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    batch_size = 1
    
    
    # Show the output
    short_target = os.path.basename(args.target_img).split('.')[0] if args.target_img else 'notarget'
    short_noise = os.path.basename(args.noise_img).split('.')[0] if args.noise_img else 'nonoise'
    short_levels = "".join(str(i) for i in args.rep_levels) if args.rep_levels else 'norep'
    save_suffix = '_'.join([args.sampler, str(args.t_total)+'t', str(args.t_start), str(args.t_end), short_levels, \
        short_noise, short_target, args.prompt, args.change_mode, 'seed'+str(args.seed), str(args.structure), args.test_suffix])
    diffusion.save_suffix = '_'.join([args.sampler, str(args.t_total)+'t', str(args.t_start), short_levels, \
        short_noise, short_target, args.prompt, args.change_mode, 'seed'+str(args.seed), str(args.structure), args.test_suffix])


    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    ###############
    # Revise by Jac
    laprep = None
    if args.target_img:
        target = img2tensor(Image.open(args.target_img).convert('RGB').resize((64, 64)))
        images = []
        for i in range(args.t_total):
            image = target.expand((batch_size * 2, 3, 64, 64)).cuda()
            step = th.Tensor([[i]]).long()
            image = diffusion.q_sample(target, step).expand((batch_size * 2, 3, 64, 64)).cuda()
            images.append(image)
#             print(i, image.max(), image.min())
        laprep = images

        diffusion.target_np = target[0].permute(1, 2, 0).cpu().detach().numpy()
        diffusion.target_structure = diffusion.target_np
        diffusion.target_structure = cv2.bilateralFilter(diffusion.target_np, \
                                                         args.rep_levels[0]*6, args.rep_levels[-1], args.rep_levels[0])
        diffusion.target_structure *= args.structure
        diffusion.target_structure_mask = np.asarray(Image.open('dataset/mouse_edge_mask.png').convert('RGB').resize((64, 64)))
        diffusion.target_structure_mask = diffusion.target_structure_mask.astype('float')/255

    noise = None
    if args.noise_img:
        if args.sampler == 'ddim_reverse':
            noise = img2tensor(Image.open(args.noise_img).convert('RGB').resize((64, 64)))
        elif args.sampler == 'ddim':
            noise = transforms.ToTensor()(np.load(args.noise_img)[0]).unsqueeze(0).float()
        else:
            print('noise_img not implemented for the sampler', args.sampler)
        noise = noise.expand((batch_size * 2, 3, 64, 64)).cuda()

    diffusion.num_timesteps = args.t_total - 1
    diffusion.args = args

    ##############################
    # Sample from the base model #
    ##############################
    upsample_temp = 0.997
    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(args.prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    # Sample from the base model.
    model.del_cache()
    if 'ddim' in args.sampler:
        samples = diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            laprep=laprep,
            reverse=bool(args.sampler=='ddim_reverse')
        )[:batch_size]
    elif 'ddpm' in args.sampler:
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            laprep=laprep,
        )[:batch_size]
    model.del_cache()

    show_images(args, samples, suffix=save_suffix)
    if args.sampler=='ddim_reverse':
        samples_np = samples.permute(0, 2, 3, 1).detach().cpu()
        np.save('dataset/'+short_noise+'_ddim_reverse.npy', samples_np)


    ##############################
    # Upsample the 64x64 samples #
    ##############################
    if args.sampler!='ddim_reverse':
        diffusion_up.args = None
        tokens = model_up.tokenizer.encode(args.prompt)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()

        # Show the output
        show_images(args, up_samples, suffix=save_suffix)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--noise_img', type=str, default=None)
    parser.add_argument('--target_img', type=str, default=None)
    parser.add_argument('--test_suffix', default='const1', help='Should include info about schedule_fn')
    parser.add_argument('--t_total', type=int, default=100)
    parser.add_argument('--t_start', type=int, default=100)
    parser.add_argument('--t_end', type=int, default=50)
    parser.add_argument('--rep_levels', type=str, default='0,1,2')
    parser.add_argument('--change_mode', type=str, default='mean', help='must be in {mean, xt}')
    parser.add_argument('--direction', type=str, default='backward', help='must be in {forward, backward}')
    parser.add_argument('--sampler', type=str, default='ddpm', help='must be in {ddim, ddpm, ddim_reverse}')
    parser.add_argument('--structure', type=float, default=1.0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    # Create base model.
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = str(args.t_total) # use 100 diffusion steps for fast sampling

    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    value_stds = [0.3] #0.05, 0.2, 0.1, 0.4, 
    t_start=[78]
    num_swap = [1,2,3,4,5,6,7,8,9,10,15,20,40] #1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,[75,80,85,90,95]
    diffusion.sturcture_diff = []
    for kernel in [5]:
        # try different spatial std
        for l in range(len(value_stds)):
            # try different value std
            args.rep_levels = [kernel, kernel, value_stds[l]]
            for s in t_start:
                args.t_start = s
                for e in num_swap:
                    if s - e>0:
                        args.t_end = s-e
                        if args.direction == 'forward':
                            assert args.tarsget_img is None
                            args.t_start = -1
                            args.t_end = -1
                            args.rep_levels = None

                        print(args.rep_levels, args.t_start, args.t_end)
                        args.schedule_fn = lambda x: int(x<=args.t_start and x>args.t_end)
                        diffusion.flag = False
                        main(args, model, diffusion, model_up, diffusion_up)    