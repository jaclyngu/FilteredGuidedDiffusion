from apy import *
from apy.amedia import *
import torchvision.transforms as transforms
import torch as th
from fgdex import *
# sys.path.append(os.path.abspath('../'))
# import StableDiffusion
import Text2Im
from apy.utils import *
import Img2Img
import PnP

import numbers

class SDFGDExperiment(FGDExperiment):
    def __init__(self, path=None, **kwargs):
        super(SDFGDExperiment, self).__init__(path=path, **kwargs);
        self.mask_strength = 1.0;
        self.weights_used = [];
        self.alphas = [];
        self.use_experimentfunc = False;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_FGD = True
        self.exe_time = -1

    def Filter(self, sampler, sampleim, meanim, use_original_steps, t):
        if self.gaussian == 'pytorch':
            #only use this if you want to use pytorch version
            return self.gaussian_adaptive_filter_pytorch(sampler, sampleim, meanim, use_original_steps, t)
        else:
            return self.adaptive_filter(sampler, sampleim, meanim, use_original_steps, t)
#             return self.bilateral_filter_workexceptmask(sampler, sampleim, meanim, use_original_steps, t)
    

    def FilterX0(self, sampler, pred_x0, added, use_original_steps, t):
        return self.adaptive_filter_x0(sampler, pred_x0, added, use_original_steps, t)
    
    
    def adaptive_filter_x0(self, sampler, pred_x0, added, use_original_steps, t):
        self.timesteps.append(t);
        [noise, a_prev, sigma_t, e_t, sqrt_one_minus_at, at_sqrt] = added
        
        bs, ch, h, w = pred_x0.size()
        meanim = pred_x0
        meanim = meanim[0]
        # 4x1x4096
        meanim_structure = meanim.reshape((ch, 1, h*w))
        # 4x1x4096
        meanim_structure = torch.bmm(meanim_structure, self.guide_image.jbm)
        meanim_structure = meanim_structure.reshape((ch, h, w))

        alphas = sampler.model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        self.alphas = alphas;
        a_t =  alphas[t]
        sqrt_alpha_cumprod = a_t.sqrt()

        # if first round, set replacement strength to 0
        if(len(self.structure_residuals)<1):
            weight = 1
            weight_control = 1;
            self.weight_schedule.append(1)
        else:
            # look at the previous error
            lastError = self.structure_residuals[-1]

            # control weight scales as structure residual gets close to self.target_dstructure
            # this naming isn't great
            weight_control = np.clip(lastError / self.target_detail, 0, 1)
            weight_control = weight_control ** self.guidance_exponent
            # optionally raise to some exponent. exponent>1 will make control more lenient
            # weigh the control by the estimated signal strength
            weight = weight_control * sqrt_alpha_cumprod

            # add weight to record of schedule
            self.weight_schedule.append(weight_control)

        do_normalize = False;
        ng = self.normalize_guidance;
        if(isinstance(ng, numbers.Number)):
            if(t<=self.t_start and (self.t_start-t)<ng):
                do_normalize = "Whiten";
        
        if(do_normalize=="Whiten" or self.normalize_guidance==True):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
            if(do_normalize=="Whiten"):
                print("Whitening at step {}".format(t))
        elif(self.normalize_guidance=="STDs"):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
        else:
            target_structure = self.guide_image.latent_structure*sqrt_alpha_cumprod;
            [d_structure, weight] = self.calcDStructure(target_structure, meanim_structure, t, weight, weight_control);

        # measure structure residual
        d_structure_value = torch.norm(d_structure, p=1).item()/np.prod(d_structure.size())

        # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
        self.structure_residuals.append(d_structure_value)
        
        # new mean is mean plus weight times the change in structure
        newmean = meanim + weight * d_structure
        diff = newmean-pred_x0
        e_t += diff*at_sqrt/sqrt_one_minus_at
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        x_prev = a_prev.sqrt() * newmean + dir_xt + noise
            
        if t > self.t_start or t <= self.t_end:
            print("Skipping {}".format(t))
            return -1
        return x_prev
    
    
            
    def adaptive_filter(self, sampler, sampleim, meanim, use_original_steps, t):
        self.timesteps.append(t);
        bs, ch, h, w = meanim.size()
        meanim = sampleim
        meanim = meanim[0]
        # 4x1x4096
        meanim_structure = meanim.reshape((ch, 1, h*w))
        # 4x1x4096
        meanim_structure = torch.bmm(meanim_structure, self.guide_image.jbm)
        meanim_structure = meanim_structure.reshape((ch, h, w))

        alphas = sampler.model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        self.alphas = alphas;
        a_t =  alphas[t]
        sqrt_alpha_cumprod = a_t.sqrt()

        # if first round, set replacement strength to 0
        if(len(self.structure_residuals)<1):
            weight = 1
            weight_control = 1;
            self.weight_schedule.append(1)
        else:
            # look at the previous error
            lastError = self.structure_residuals[-1]

            # control weight scales as structure residual gets close to self.target_dstructure
            # this naming isn't great
            weight_control = np.clip(lastError / self.target_detail, 0, 1)
            weight_control = weight_control ** self.guidance_exponent
            # optionally raise to some exponent. exponent>1 will make control more lenient
            # weigh the control by the estimated signal strength
            weight = weight_control * sqrt_alpha_cumprod

            # add weight to record of schedule
            self.weight_schedule.append(weight_control)

        do_normalize = False;
        ng = self.normalize_guidance;
        if(isinstance(ng, numbers.Number)):
            if(t<=self.t_start and (self.t_start-t)<ng):
                do_normalize = "Whiten";
        
        if(do_normalize=="Whiten" or self.normalize_guidance==True):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
            if(do_normalize=="Whiten"):
                print("Whitening at step {}".format(t))
        elif(self.normalize_guidance=="STDs"):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
        else:
            target_structure = self.guide_image.latent_structure*sqrt_alpha_cumprod;
            [d_structure, weight] = self.calcDStructure(target_structure, meanim_structure, t, weight, weight_control);

        # measure structure residual
        d_structure_value = torch.norm(d_structure, p=1).item()/np.prod(d_structure.size())

        # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
        self.structure_residuals.append(d_structure_value)
        
        # new mean is mean plus weight times the change in structure
        newmean = meanim + weight * d_structure

        if t > self.t_start or t <= self.t_end:
            print("Skipping {}".format(t))
            return sampleim
        return newmean
    
    def bilateral_filter_workexceptmask(self, sampler, sampleim, meanim, use_original_steps, t):
        self.timesteps.append(t);
        bs, ch, h, w = meanim.size()
        meanim = meanim[0]

        # 4x1x4096
        meanim_structure = meanim.reshape((ch, 1, h*w))
        # 4x1x4096
        meanim_structure = torch.bmm(meanim_structure, self.guide_image.jbm)
        meanim_structure = meanim_structure.reshape((ch, h, w))

        
        alphas = sampler.model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        a_t =  alphas[t]
        sqrt_alpha_cumprod = a_t.sqrt()
        
        noiseim = sampleim - meanim
        
        # if first round, set replacement strength to 0
        if(len(self.structure_residuals)<1):
            weight = 1
            self.weight_schedule.append(1)
        else:
            # look at the previous error
            lastError = self.structure_residuals[-1]

            # control weight scales as structure residual gets close to self.target_dstructure
            # this naming isn't great
            weight_control = np.clip(lastError / self.target_detail, 0, 1)
            
            # optionally raise to some exponent. exponent>1 will make control more lenient
            weight_control = weight_control ** self.guidance_exponent
#             print('weight_control', weight_control)

            # weigh the control by the estimated signal strength
            weight = weight_control * sqrt_alpha_cumprod

            # add weight to record of schedule
            self.weight_schedule.append(weight_control)
        
        # the normalization I mentioned
        target_mean = torch.mean(self.guide_image.latent_structure, (1,2), keepdim=True)
        target_std = torch.std(self.guide_image.latent_structure, (1,2), keepdim=True)
        meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
        meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
        target_structure = (self.guide_image.latent_structure-target_mean) / target_std
        target_structure = target_structure * meanim_std + meanim_mean
        # calculate the difference in structure
        d_structure = target_structure - meanim_structure

        # add diff in structure
#         self.d_structures.append(d_structure)

        # measure structure residual
        d_structure_value = torch.norm(d_structure, p=1).item()/np.prod(d_structure.size())

        # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
        self.structure_residuals.append(d_structure_value)

        # new mean is mean plus weight times the change in structure
        newmean = meanim + weight * d_structure
        
#         newmean_structure = torch.bmm(newmean.reshape((ch, 1, h*w)), self.cbm).reshape((ch, h, w))
#         newmean_detail = newmean - newmean_structure
#         xt_sturcture = torch.bmm(self.forward_xt[t].reshape((ch, 1, h*w)), self.cbm).reshape((ch, h, w))
#         xt_detail = self.forward_xt[t][0] - xt_sturcture
#         detail_c = torch.norm(xt_detail, p=2, dim=(1,2), keepdim=True)/torch.norm(newmean_detail, p=2, dim=(1,2), keepdim=True)
#         print(detail_c.flatten(), t, len(self.forward_xt))
#         newmean_detail = newmean_detail*detail_c
#         newmean = newmean_structure + newmean_detail
        
        if t > self.t_start or t <= self.t_end:
            return sampleim
        print('swapped!', weight, t)
        return newmean + noiseim
    
    
    def gaussian_adaptive_filter_pytorch(self, sampler, sampleim, meanim, use_original_steps, t):
        self.timesteps.append(t);
        bs, ch, h, w = meanim.size()
        meanim = sampleim
        meanim = meanim[0]
        # 4x1x4096
        meanim_structure = meanim#.reshape((ch, 1, h*w))
        # 4x1x4096  
        
        meanim_structure = self.gaussian_filter(meanim_structure)
#         meanim_structure = meanim_structure.reshape((ch, h, w))

        alphas = sampler.model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        self.alphas = alphas;
        a_t =  alphas[t]
        sqrt_alpha_cumprod = a_t.sqrt()

        # if first round, set replacement strength to 0
        if(len(self.structure_residuals)<1):
            weight = 1
            weight_control = 1;
            self.weight_schedule.append(1)
        else:
            # look at the previous error
            lastError = self.structure_residuals[-1]

            # control weight scales as structure residual gets close to self.target_dstructure
            # this naming isn't great
            weight_control = np.clip(lastError / self.target_detail, 0, 1)
            weight_control = weight_control ** self.guidance_exponent
            # optionally raise to some exponent. exponent>1 will make control more lenient
            # weigh the control by the estimated signal strength
            weight = weight_control * sqrt_alpha_cumprod

            # add weight to record of schedule
            self.weight_schedule.append(weight_control)

        do_normalize = False;
        ng = self.normalize_guidance;
        if(isinstance(ng, numbers.Number)):
            if(t<=self.t_start and (self.t_start-t)<ng):
                do_normalize = "Whiten";
        
        if(do_normalize=="Whiten" or self.normalize_guidance==True):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
            if(do_normalize=="Whiten"):
                print("Whitening at step {}".format(t))
        elif(self.normalize_guidance=="STDs"):
            latent_target_structure = self.guide_image.latent_structure;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            target_std = torch.std(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            meanim_std = torch.std(meanim_structure, (1,2), keepdim=True)
            target_structure = (latent_target_structure-target_mean) / target_std
            target_structure = target_structure * meanim_std + meanim_mean
            d_structure = target_structure - meanim_structure
        else:
            target_structure = self.guide_image.latent_structure*sqrt_alpha_cumprod;
            [d_structure, weight] = self.calcDStructure(target_structure, meanim_structure, t, weight, weight_control);

        # measure structure residual
        d_structure_value = torch.norm(d_structure, p=1).item()/np.prod(d_structure.size())

        # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
        self.structure_residuals.append(d_structure_value) 
        
        # new mean is mean plus weight times the change in structure
        newmean = meanim + weight * d_structure

        if t > self.t_start or t <= self.t_end:
            print("Skipping {}".format(t))
            return sampleim 
        return newmean
    

            
    @property
    def xT_latent(self):
        return self.getInfo("xT_latent");

    @xT_latent.setter
    def xT_latent(self, value):
        self.setInfo('xT_latent', value);
        

    
    # <editor-fold desc="Property: 'normalize'">
    @property
    def normalize_guidance(self):
        return self.getInfo("normalize_guidance");
    @normalize_guidance.setter
    def normalize_guidance(self, value):
        self.setInfo('normalize_guidance', value);
    # </editor-fold>
    
    # <editor-fold desc="Property: 'guidance_exponent'">
    @property
    def guidance_exponent(self):
        return self.getInfo("guidance_exponent");
    @guidance_exponent.setter
    def guidance_exponent(self, value):
        self.setInfo('guidance_exponent', value);
    # </editor-fold>
        
    
    def Evaluate(self):
        if self.xT_latent is not None and (not isinstance(self.xT_latent, str)):
            self.xT_latent = self.xT_latent.to(self.device)
        return Text2Im.Text2Im(
            experiment=self,
            **self.extra_args
        );
    
    def EvaluateSDEdit(self):
        if self.xT_latent is not None and (not isinstance(self.xT_latent, str)):
            self.xT_latent = self.xT_latent.to(self.device)
        return Img2Img.Img2Img(
            experiment=self,
            **self.extra_args
        );
    
    def EvaluatePnP(self):
        if (self.xT_latent is not None) and (not isinstance(self.xT_latent, str)):
            self.xT_latent = self.xT_latent.to(self.device)
        return PnP.PnP(
            experiment=self,
            **self.extra_args
        );
    
    
    def result_png_path(self, result_dir):
        return os.path.join(result_dir, self.GetParamString(**self.getParamDict())+".png")
    
    def plot_png_path(self, result_dir):
        return os.path.join(result_dir, self.GetParamString(**self.getParamDict())+"_plot.png")
    
    def WriteResults(self, result_dir):
        Image(pixels=self.grid_results[0]).writeToFile(self.result_png_path(result_dir))
        #Image(pixels=self.grid_results[0]).show()
        if self.use_FGD:
            self.showdStructureValues()
            plt.savefig(os.path.join(result_dir, self.GetParamString(**self.getParamDict())+"_plot.pdf"))
            plt.savefig(self.plot_png_path(result_dir))
            return Image(self.plot_png_path(result_dir));
        
    
    def getParamDict(self):
        d = super(SDFGDExperiment,self).getParamDict();
        d["normalize_guidance"]=self.normalize_guidance;
        d["guidance_exponent"] = self.guidance_exponent;
        d["guide_image"] = self.guide_image
        return d;
    
    
    def setParameters(self,
                      normalize_guidance=True,
                      guidance_exponent=1.0,
                      xT_latent=None,
                      use_FGD=True,
                      gaussian='',
                      invert_portion=1.0,
                      **kwargs):
        super(SDFGDExperiment, self).setParameters(**kwargs);
        self.normalize_guidance = normalize_guidance;
        self.guidance_exponent=guidance_exponent;
        self.xT_latent = xT_latent
        self.use_FGD=use_FGD
        self.gaussian=gaussian
        self.invert_portion=invert_portion
        
    
    @classmethod
    def GetParamString(cls,
                       guide_image=None,
                       guide_text='',
                       sigmas=[5,5,0.1],
                       t_total=100,
                       t_start=100,
                       t_end=50,
                       seed=48,
                       guidance_scale=0.0,
                       sampler='ddpm',
                       target_detail=0.4,
                       normalize_guidance=True,
                       guidance_exponent=1.0,
                       **kwargs):
        print(kwargs)
        return "{}_{}_p-{}_s-{}_sigx{:.2f}y{:.2f}v{:.3f}_tse-{}-{}-{})_dtail{}_norm{}_s{}_ge{}".format(
            cls.__name__,
            guide_image.file_name_base,
            guide_text,
            seed,
            sigmas[0],
            sigmas[1],
            sigmas[2],
            t_total,
            t_start,
            t_end,
            target_detail,
            normalize_guidance,
            guidance_exponent,
            sampler
        )
    
    def showdStructureValues(self):
        ts =np.array(self.timesteps);
        plt.figure(figsize=[5,5])
        ax1 = plt.subplot(211)
        plot = plt.plot(
            ts[:len(self.structure_residuals)], self.structure_residuals
        );
        plt.xlim(ts.max(), ts.min())
        ax2 = plt.subplot(212)
        plt.plot(ts[:len(self.weight_schedule)], self.weight_schedule);
        plt.xlim(ts.max(), ts.min())
        ax1.set_ylabel("Residuals")
        ax2.set_ylabel("$\lambda$")
        plt.suptitle("{} | $\gamma$: {}".format(self.guide_text, self.target_detail))
    
    
    def calcDStructure(self, target_structure, meanim_structure, t, weight, weight_control):
        #d_structure = target_structure - meanim_structure
#         scale = self.alphas[t].sqrt();
        scale=1.0;
        d_structure = target_structure*scale - meanim_structure;
        return [d_structure, weight];
        
        
        if(self.normalize_guidance is True):
            d_structure = target_structure - meanim_structure;
        else:
            scale = self.alphas[t].sqrt();
            latent_target_structure = self.guide_image.latent_structure*scale;
            target_mean = torch.mean(latent_target_structure, (1,2), keepdim=True)
            meanim_mean = torch.mean(meanim_structure, (1,2), keepdim=True)
            latent_target_structure = (latent_target_structure-target_mean+meanim_mean);
            d_structure = latent_target_structure - meanim_structure;


        return [d_structure, weight];
    
    
    def setGuideLatent(self, model):
        if self.guide_image:
            if(self.guide_image.latent_im is None):
                init_image = Text2Im.load_img(self.guide_image.absolute_file_path).to(self.device)
                #convert the guidance image to latent space
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                # _, ch, h, w = init_latent.size()
                self.guide_image.latent_dim = init_latent.size();
                [_, ch, h, w] = self.guide_image.latent_dim;
                init_latent = init_latent.reshape((ch, 1, h*w))
                self.guide_image.latent_im = init_latent;
                #1x4x64x64
                print('guidance image latent feature', init_latent.size())
            else:
                [_, ch, h, w] = self.guide_image.latent_dim;
                init_latent = self.guide_image.latent_im;

            if self.gaussian == 'pytorch':
                init_latent = init_latent.reshape((ch, h, w))
                self.gaussian_filter = transforms.GaussianBlur((self.sigmas[0], self.sigmas[1]), sigma=self.sigmas[2])
                self.guide_image.latent_structure = self.gaussian_filter(init_latent)
            else:
                if self.gaussian == 'acimops':
                    if(self.guide_image.jbmsigmas == self.sigmas):
                        print("Reusing JBM for sigmas {}".format(self.sigmas));
                    else:
                        cbm = self.guide_image.getGaussianCirculantMatrix(self.sigmas);
                        self.guide_image.setJBM(torch.Tensor(cbm).unsqueeze(0).repeat((ch, 1,1)).to(self.device), self.sigmas);
                
                else:    
                    # _, ch, h, w = init_latent.size()
                    #cross bilat kernel, size 4x4096
#                     if(self.guide_image.jbmsigmas == self.sigmas):
#                         print("Reusing JBM for sigmas {}".format(self.sigmas));
                    cbm = self.guide_image.getCrossBilateralMatrix(self.sigmas);
                    self.guide_image.setJBM(torch.Tensor(cbm).unsqueeze(0).repeat((ch, 1,1)).to(self.device), self.sigmas);
                    # sampler.target_structure = torch.bmm(init_latent, sampler.cbm).reshape((ch, h, w))

                if(self.guide_image.latent_structure is None):
                    init_latent = init_latent.reshape((ch, 1, h*w))
                    self.guide_image.latent_structure = torch.bmm(init_latent, self.guide_image.jbm).reshape((ch, h, w))
                
                