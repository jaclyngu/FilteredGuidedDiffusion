from apy import *
from apy.amedia import *
import torchvision.transforms as transforms
import torch as th
from fgdex import *
# sys.path.append(os.path.abspath('../'))
# import StableDiffusion
import Text2Im_ILVR
from apy.utils import *
import Img2Img
import PnP
from resizer import Resizer
import numbers

class ILVR_FGDExperiment(FGDExperiment):
    def __init__(self, path=None, **kwargs):
        super(ILVR_FGDExperiment, self).__init__(path=path, **kwargs);
        self.mask_strength = 1.0;
        self.weights_used = [];
        self.alphas = [];
        self.use_experimentfunc = False;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_FGD = True
        self.exe_time = -1

    def Filter(self, sampler, sampleim, guide_noise, use_original_steps, t):
        if t <= self.t_end:
            print("Skipping {}".format(t))
            return sampleim 
#         import pdb;pdb.set_trace()
        sampleim = sampleim - self.up(self.down(sampleim)) + self.up(self.down(guide_noise))
        return sampleim 
            
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
        return Text2Im_ILVR.Text2Im(
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
#         if self.use_FGD:
#             self.showdStructureValues()
#             plt.savefig(os.path.join(result_dir, self.GetParamString(**self.getParamDict())+"_plot.pdf"))
#             plt.savefig(self.plot_png_path(result_dir))
#             return Image(self.plot_png_path(result_dir));
        
    
    def getParamDict(self):
        d = super(ILVR_FGDExperiment,self).getParamDict();
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
                      down_N=0,
                      **kwargs):
        super(ILVR_FGDExperiment, self).setParameters(**kwargs);
        self.normalize_guidance = normalize_guidance;
        self.guidance_exponent=guidance_exponent;
        self.xT_latent = xT_latent
        self.use_FGD=use_FGD
        self.gaussian=gaussian
        self.invert_portion=invert_portion
        
        shape = (1, 4, 64, 64)
        shape_d = (1, 4, int(64 / down_N), int(64 / down_N))
        self.down = Resizer(shape, 1 / down_N).to(self.device)
        self.up = Resizer(shape_d, down_N).to(self.device)

    
        
    
    @classmethod
    def GetParamString(cls,
                       guide_image=None,
                       guide_text='',
                       sigmas=[5,5,0.1],
                       t_total=100,
                       t_start=100,
                       t_end=50,
                       seed=2,
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
                init_image = Text2Im_ILVR.load_img(self.guide_image.absolute_file_path).to(self.device)
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
                    if(self.guide_image.jbmsigmas == self.sigmas):
                        print("Reusing JBM for sigmas {}".format(self.sigmas));
                    else:
                        cbm = self.guide_image.getCrossBilateralMatrix(self.sigmas);
                        self.guide_image.setJBM(torch.Tensor(cbm).unsqueeze(0).repeat((ch, 1,1)).to(self.device), self.sigmas);
                        # sampler.target_structure = torch.bmm(init_latent, sampler.cbm).reshape((ch, h, w))

                if(self.guide_image.latent_structure is None):
                    init_latent = init_latent.reshape((ch, 1, h*w))
                    self.guide_image.latent_structure = torch.bmm(init_latent, self.guide_image.jbm).reshape((ch, h, w))
                
                