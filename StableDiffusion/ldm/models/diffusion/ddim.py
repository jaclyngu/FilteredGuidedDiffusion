"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

#     @torch.no_grad()
#     def sample(self,
#                S,
#                batch_size,
#                shape,
#                conditioning=None,
#                callback=None,
#                normals_sequence=None,
#                img_callback=None,
#                quantize_x0=False,
#                eta=0.,
#                mask=None,
#                x0=None,
#                temperature=1.,
#                noise_dropout=0.,
#                score_corrector=None,
#                corrector_kwargs=None,
#                verbose=True,
#                x_T=None,
#                log_every_t=100,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
#                **kwargs
#                ):
#         if conditioning is not None:
#             if isinstance(conditioning, dict):
#                 cbs = conditioning[list(conditioning.keys())[0]].shape[0]
#                 if cbs != batch_size:
#                     print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
#             else:
#                 if conditioning.shape[0] != batch_size:
#                     print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

#         self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
#         # sampling
#         C, H, W = shape
#         size = (batch_size, C, H, W)
#         print(f'Data shape for DDIM sampling is {size}, eta {eta}')

#         samples, intermediates = self.ddim_sampling(conditioning, size,
#                                                     callback=callback,
#                                                     img_callback=img_callback,
#                                                     quantize_denoised=quantize_x0,
#                                                     mask=mask, x0=x0,
#                                                     ddim_use_original_steps=False,
#                                                     noise_dropout=noise_dropout,
#                                                     temperature=temperature,
#                                                     score_corrector=score_corrector,
#                                                     corrector_kwargs=corrector_kwargs,
#                                                     x_T=x_T,
#                                                     log_every_t=log_every_t,
#                                                     unconditional_guidance_scale=unconditional_guidance_scale,
#                                                     unconditional_conditioning=unconditional_conditioning,
#                                                     )
#         return samples, intermediates

#     @torch.no_grad()
#     def ddim_sampling(self, cond, shape,
#                       x_T=None, ddim_use_original_steps=False,
#                       callback=None, timesteps=None, quantize_denoised=False,
#                       mask=None, x0=None, img_callback=None, log_every_t=100,
#                       temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
#                       unconditional_guidance_scale=1., unconditional_conditioning=None,):
#         device = self.model.betas.device
#         b = shape[0]
#         if x_T is None:
#             img = torch.randn(shape, device=device)
#         else:
#             img = x_T

#         if timesteps is None:
#             timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
#         elif timesteps is not None and not ddim_use_original_steps:
#             subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
#             timesteps = self.ddim_timesteps[:subset_end]

#         intermediates = {'x_inter': [img], 'pred_x0': [img]}
#         time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
#         total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
#         print(f"Running DDIM Sampling with {total_steps} timesteps")

#         iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

#         for i, step in enumerate(iterator):
#             index = total_steps - i - 1
#             ts = torch.full((b,), step, device=device, dtype=torch.long)

#             if mask is not None:
#                 assert x0 is not None
#                 img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
#                 img = img_orig * mask + (1. - mask) * img

#             outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
#                                       quantize_denoised=quantize_denoised, temperature=temperature,
#                                       noise_dropout=noise_dropout, score_corrector=score_corrector,
#                                       corrector_kwargs=corrector_kwargs,
#                                       unconditional_guidance_scale=unconditional_guidance_scale,
#                                       unconditional_conditioning=unconditional_conditioning)
#             img, pred_x0, pseudo_mean = outs
#             if callback: callback(i)
#             if img_callback: img_callback(pred_x0, i)

#             if index % log_every_t == 0 or index == total_steps - 1:
#                 intermediates['x_inter'].append(img)
#                 intermediates['pred_x0'].append(pred_x0)

#         return img, intermediates

#     def bilateral_filter_workexceptmask(self, sampleim, meanim, coeff, use_original_steps, t=-1):

#         if coeff == 0 and self.opt.dont_record:
#             return sampleim
#         bs, ch, h, w = meanim.size()
#         meanim = meanim[0]
#         # 4x1x4096
#         meanim_structure = meanim.reshape((ch, 1, h*w))
#         # 4x1x4096
#         meanim_structure = torch.bmm(meanim_structure, self.cbm)
#         meanim_structure = meanim_structure.reshape((ch, h, w))

#         alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
#         a_t =  alphas[t]
#         sqrt_alpha_cumprod = a_t.sqrt()

#         noiseim = sampleim - meanim

#         # if first round, set replacement strength to 0
#         if(len(self.d_structure_values)<1):
#             weight = 1
#             self.weight_schedule.append(1)
#         else:
#             # look at the previous error
#             lastError = self.d_structure_values[-1]

#             # control weight scales as structure residual gets close to self.target_dstructure
#             # this naming isn't great
#             weight_control = np.clip(lastError / self.opt.structure, 0, 1)

#             # optionally raise to some exponent. exponent>1 will make control more lenient
#             weight_control = weight_control ** self.opt.target_control_exponent
# #             print('weight_control', weight_control)

#             # weigh the control by the estimated signal strength
#             weight = weight_control * sqrt_alpha_cumprod

#             # add weight to record of schedule
#             self.weight_schedule.append(weight_control)

#         # the normalization I mentioned
#         target_mean = torch.mean(self.target_structure, (1,2), keepdim=True)
#         target_std = torch.std(self.target_structure, (1,2), keepdim=True)
#         meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
#         meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
#         target_structure = (self.target_structure-target_mean) / target_std
#         target_structure = target_structure * meanim_std + meanim_mean
#         # calculate the difference in structure
#         d_structure = target_structure - meanim_structure

#         # add diff in structure
# #         self.d_structures.append(d_structure)

#         # measure structure residual
#         d_structure_value = torch.norm(d_structure, p=1).item()/np.prod(d_structure.size())

#         # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
#         self.d_structure_values.append(d_structure_value)

#         # new mean is mean plus weight times the change in structure
#         newmean = meanim + weight * d_structure

# #         newmean_structure = torch.bmm(newmean.reshape((ch, 1, h*w)), self.cbm).reshape((ch, h, w))
# #         newmean_detail = newmean - newmean_structure
# #         xt_sturcture = torch.bmm(self.forward_xt[t].reshape((ch, 1, h*w)), self.cbm).reshape((ch, h, w))
# #         xt_detail = self.forward_xt[t][0] - xt_sturcture
# #         detail_c = torch.norm(xt_detail, p=2, dim=(1,2), keepdim=True)/torch.norm(newmean_detail, p=2, dim=(1,2), keepdim=True)
# #         print(detail_c.flatten(), t, len(self.forward_xt))
# #         newmean_detail = newmean_detail*detail_c
# #         newmean = newmean_structure + newmean_detail

#         if coeff == 0:
#             return sampleim
#         print('swapped!', weight, t)
#         return newmean + noiseim


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        ###############
        # Revise by Jac: additionally return pseudo_mean
        ###############
        pseudo_mean = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0, pseudo_mean


    @torch.no_grad()
    def encode_ddim(self, img, num_steps,conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1., invert_portion=None):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        T = 999
        c = T // num_steps
        iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
        steps = list(range(0,T + c,c))

        for i, t in enumerate(iterator):
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)

        return img, _

    @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod #.flip(0)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0 

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        img = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            img, pred_x0, pseudo_mean = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)

            ###############
            # Revise by Jac
            ###############
            
            FGD = self.opt.get("experiment");
            if FGD.use_FGD:
                print("DDIM+FGD decode func!")
                with torch.no_grad():
                    img = FGD.Filter(self, img, pseudo_mean, use_original_steps, index).expand(*pred_x0.size())

            ###############
        return img