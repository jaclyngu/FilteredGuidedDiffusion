import torch
from cbilateral import getCrossBilateralMatrix4D
import json
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
class FGD():
    def __init__(self, diffusionModel, guide_image, detail=1.2, sigmas=[3,3,0.3], t_end=15, norm_steps=0):
        self.guide_image = guide_image
        self.detail = detail
        self.t_end = t_end
        self.sigmas = sigmas
        self.norm_steps = norm_steps
        self.model = diffusionModel
        self.bilateral_matrix_4d = None
        
        self.guide_latent = None
        self.guide_structure = None
        self.guide_structure_normalized = None

        self.init_guide_latent = None
        self.init_guide_structure = None
        self.init_guide_stucture_normalized=None
        
        self.init_bilateral_matrix_4d=None
        self.set_guide_image(guide_image)

    def set_ST(self, detail=1.6, recompute_matrix=True, sigmas=[3,3,0.3]):
        if recompute_matrix:
            self.set_bilateral_matrix(sigmas)
        self.detail = detail
        self.t_end = 15
        self.norm_steps = 50


    def reset(self):
        self.init_guide_latent = self.guide_latent
        self.guide_structure = self.init_guide_structure
        self.guide_structure_normalized = self.init_guide_structure_normalized
        self.bilateral_matrix_4d = self.init_bilateral_matrix_4d

    def set_guide_image(self, guide_image):
        self.guide_latent = self.model.encode_image(guide_image)
        self.guide_image = guide_image
        if self.sigmas != None:
            self.set_bilateral_matrix(self.sigmas)

    def set_bilateral_matrix(self,sigmas):
        assert len(sigmas)==2 or len(sigmas)==3, "sigmas has invalid number of entries (either 2 or 3)"
        sigmas = np.array(sigmas).astype(np.double)
        if len(sigmas) == 2:
            sigmas = np.insert(sigmas, 1, sigmas[0])

        guide_latent_processed = self.guide_latent.detach().cpu().permute(0, 2, 3, 1).numpy()
        guide_latent_processed = np.squeeze(guide_latent_processed)
        bilateral_matrix = getCrossBilateralMatrix4D(guide_latent_processed.astype('double'),sigmas)
        self.bilateral_matrix_4d = torch.Tensor(bilateral_matrix).unsqueeze(0).repeat((4,1,1)).to(device)
        guide_structure_latent = torch.matmul(self.bilateral_matrix_4d, self.guide_latent.reshape(4,4096,1))
        guide_structure_latent = guide_structure_latent.reshape(1,4,64,64)

        guide_mean = torch.mean(guide_structure_latent, (2,3), keepdim=True)
        guide_std = torch.std(guide_structure_latent, (2,3), keepdim=True)

        self.guide_structure_normalized = (guide_structure_latent - guide_mean) / guide_std
        self.guide_structure = guide_structure_latent

        self.init_guide_structure = self.guide_structure
        self.init_guide_structure_normalized=self.guide_structure_normalized
        self.init_bilateral_matrix_4d = self.bilateral_matrix_4d

        self.sigmas = sigmas.tolist()
    
    def get_residual_structure(self, latents):
        current_structure = torch.matmul(self.bilateral_matrix_4d, latents.reshape(4,4096,1))
        current_structure = current_structure.reshape(1,4,64,64)

        d_structure = self.guide_structure - current_structure
        return d_structure
    
    def get_structure(self, latents, bm_4d=None):
        if bm_4d ==None:
            bm_4d = self.bilateral_matrix_4d
        structure = torch.matmul(bm_4d, latents.reshape(4,4096,1))
        structure = structure.reshape(1,4,64,64)
        return structure

    def get_guidance(self, latents, input_latents, scheduler, t):
        guide_low = self.guide_structure
        
        st_low = self.get_structure(latents)
        st_high = latents - st_low

        weight= self.detail

        d = guide_low - st_low
        
        return weight, d

    
    def get_guidance_normalized(self, latents, input_latents, scheduler, t):
        current_structure = self.get_structure(latents)
        guide_structure = self.guide_structure
            
        current_mean = torch.mean(current_structure, (2,3), keepdim=True)
        current_std = torch.std(current_structure, (2,3), keepdim=True)

        guide_structure_renormalized = self.guide_structure_normalized * current_std + current_mean
        d_structure_renormalized = guide_structure_renormalized - current_structure

        residual_score = torch.mean(torch.abs(d_structure_renormalized)) 

        weight = self.detail

        return weight, d_structure_renormalized
       
    def get_params(self):
        params = {
            'guide image':self.guide_image,
            'detail':self.detail,
            'sigmas':self.sigmas,
            't_end':self.t_end,
            'norm steps':self.norm_steps,
        }
        return params
    def __str__(self):
        return (json.dumps(self.get_params(), indent=2))