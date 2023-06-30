import os
import hashlib
from apy.aobject.SavesFeatures import *
from apy.aobject.SavesDirectories import *
from apy.utils import make_sure_dir_exists
from apy.amedia import *
from .GuideImage import *

import torchvision.transforms as transforms


def tensor2img(tensor):
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu()
    # tensor = tensor.permute(1, 2, 0).detach().cpu()
    #     tensor = tensor.permute(0, 2, 3, 1).detach().cpu()
    return Image(pixels=tensor.numpy())
#     return Image(pixels=(tensor.numpy()[0]+1)/2);

def img2tensor(img):
    # img_tensor = transforms.ToTensor()(img).unsqueeze(0).float()
    img_tensor = transforms.ToTensor()(img.fpixels).unsqueeze(0).float()
    #     img_tensor = img_tensor * 2 -1
    return img_tensor

class FGDExperiment(SavesFeatures, SavesDirectories):
    def __init__(self, path=None, **kwargs):
        self.results = []
        self.grid_results = [];
        self.upsampled_results = [];
        self._guide_image_scaled = None;
        self.means = [];
        super(FGDExperiment, self).__init__(path=path, **kwargs);
        if(self.structure_residuals is None):
            self.structure_residuals = [];
        if(self.weight_schedule is None):
            self.weight_schedule = [];
        if(self.timesteps is None):
            self.timesteps = [];
        if(self.sqrt_alpha_cumprods is None):
            self.sqrt_alpha_cumprods = [];
        if(self.d_structure_values is None):
            self.d_structure_values = [];
        if(self.use_original_steps is None):
            self.use_original_steps = False;

        # guide_image=guide_image,
        # t_total=t_total,
        # t_start=t_start,
        # t_end=t_end,
        # spatial_std=spatial_std,
        # value_std=value_std,
        # target_detail=target_detail

        # self.guide_scale=[64, 64];

    def GetDescription(self):
        return '''
        This version of the experiment works like ___
        ''';


    def setParameters(self,
                      guide_image=None,
                      guide_text='',
                      sigmas=[5,5,0.1],
                      t_total=100,
                      t_start=100,
                      t_end=50,
                      seed=2,
                      guidance_scale=3.0,
                      target_detail=0.4,
                      sampler='ddpm',
                      use_original_steps=False,
                      **kwargs):
        self.guide_image = guide_image;
        self.guide_text = guide_text;
        self.sigmas = sigmas;
        self.t_total = t_total;
        self.t_start = t_start;
        self.t_end = t_end;
        self.seed = seed;
        self.guidance_scale=guidance_scale;
        self.sampler=sampler;
        self.extra_args = kwargs;
        self.description = self.GetDescription()
        self.target_detail=target_detail;
        self.use_original_steps = use_original_steps


    def getParamDict(self):
        return dict(
            guide_image=self.guide_image,
            guide_text=self.guide_text,
            sigmas=self.sigmas,
            t_total=self.t_total,
            t_start=self.t_start,
            t_end=self.t_end,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            sampler=self.sampler,
            target_detail = self.target_detail,
            use_original_steps = self.use_original_steps,
            **self.extra_args
        )

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
                       **kwargs):
        return "{}_{}_p-{}_s-{}_sigx{:.2f}y{:.2f}v{:.3f}_tse-{}-{}-{})_dtail{}_gs{}_s{}".format(
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
            guidance_scale,
            sampler
        )

    @classmethod
    def GetDirNameForParams(cls,
                            **kwargs):
        return cls.GetParamString(**kwargs)

    def getParamString(self):
        return FGDExperiment.GetParamString(**self.getParamDict())

    def saveResultTensors(self, samples, upsamples):
        samplesnp = samples.permute(0, 2, 3, 1).detach().cpu().numpy()[0];
        upsamplesnp = upsamples.permute(0, 2, 3, 1).detach().cpu().numpy()[0];
        result = Image(pixels=(samplesnp+1)*0.5)
        upresult = Image(pixels=(upsamplesnp+1)*0.5)
        result.writeToFile(os.path.join(self.base_dir, self.getParamString()+".png"));
        upresult.writeToFile(os.path.join(self.base_dir, "upsampled_"+self.getParamString()+".png"));
        self.results.append([result, upresult]);

    @property
    def base_dir(self):
        return self.getDirectoryPath()+os.sep;


    @classmethod
    def Create(cls, directory,
               guide_image=None,
               guide_text='',
               target_detail=0.4,
               t_end=50,
               sigmas=[5,5,0.1],
               t_total=100,
               t_start=100,
               seed=2,
               sampler='ddpm',
               **kwargs):
        args = dict(
            guide_image=guide_image,
            guide_text=guide_text,
            sigmas=sigmas,
            t_total=t_total,
            t_start=t_start,
            t_end=t_end,
            seed=seed,
            target_detail=target_detail,
            sampler=sampler,
            **kwargs
        )
        dname = cls.GetDirNameForParams(**args);
        pathdir = os.path.join(directory, dname)+os.sep;
        make_sure_dir_exists(pathdir);
        path = os.path.join(pathdir, cls.__name__+'.json');
        newInstance = cls(path=path);
        newInstance.setParameters(**args);
        newInstance.saveJSON();
        return newInstance;


    # <editor-fold desc="Property: 'description'">
    @property
    def description(self):
        return self.getInfo("description");
    @description.setter
    def description(self, value):
        self.setInfo('description', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'extra_args'">
    @property
    def extra_args(self):
        return self.getInfo("extra_args");
    @extra_args.setter
    def extra_args(self, value):
        self.setInfo('extra_args', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'sigmas'">
    @property
    def sigmas(self):
        return self.getInfo("sigmas");
    @sigmas.setter
    def sigmas(self, value):
        self.setInfo('sigmas', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'seed'">
    @property
    def seed(self):
        return self.getInfo("seed");
    @seed.setter
    def seed(self, value):
        self.setInfo('seed', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'use_original_steps'">
    @property
    def use_original_steps(self):
        return self.getInfo("use_original_steps");
    @use_original_steps.setter
    def use_original_steps(self, value):
        self.setInfo('use_original_steps', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'guide_text'">
    @property
    def guide_text(self):
        return self.getInfo("guide_text");
    @guide_text.setter
    def guide_text(self, value):
        self.setInfo('guide_text', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'guidance_scale'">
    @property
    def guidance_scale(self):
        return self.getInfo("guidance_scale");
    @guidance_scale.setter
    def guidance_scale(self, value):
        self.setInfo('guidance_scale', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'target_detail'">
    @property
    def target_detail(self):
        return self.getInfo("target_detail");
    @target_detail.setter
    def target_detail(self, value):
        self.setInfo('target_detail', value);
    # </editor-fold>



    # <editor-fold desc="Property: 't_total'">
    @property
    def t_total(self):
        return self.getInfo("t_total");
    @t_total.setter
    def t_total(self, value):
        self.setInfo('t_total', value);
    # </editor-fold>

    # <editor-fold desc="Property: 't_start'">
    @property
    def t_start(self):
        return self.getInfo("t_start");
    @t_start.setter
    def t_start(self, value):
        self.setInfo('t_start', value);
    # </editor-fold>

    # <editor-fold desc="Property: 't_end'">
    @property
    def t_end(self):
        return self.getInfo("t_end");
    @t_end.setter
    def t_end(self, value):
        self.setInfo('t_end', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'sampler'">
    @property
    def sampler(self):
        return self.getInfo("sampler");
    @sampler.setter
    def sampler(self, value):
        self.setInfo('sampler', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'sampler'">
    @property
    def sampler(self):
        return self.getInfo("sampler");
    @sampler.setter
    def sampler(self, value):
        self.setInfo('sampler', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'structure_residuals'">
    @property
    def structure_residuals(self):
        return self.getInfo("structure_residuals");
    @structure_residuals.setter
    def structure_residuals(self, value):
        self.setInfo('structure_residuals', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'timesteps'">
    @property
    def timesteps(self):
        return self.getInfo("timesteps");
    @timesteps.setter
    def timesteps(self, value):
        self.setInfo('timesteps', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'weight_schedule'">
    @property
    def weight_schedule(self):
        return self.getInfo("weight_schedule");
    @weight_schedule.setter
    def weight_schedule(self, value):
        self.setInfo('weight_schedule', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'd_structure_values'">
    @property
    def d_structure_values(self):
        return self.getInfo("d_structure_values");
    @d_structure_values.setter
    def d_structure_values(self, value):
        self.setInfo('d_structure_values', value);
    # </editor-fold>

    @property
    def guide_image_path(self):
        return os.path.join(self.getDirectoryPath(), "guide_image.png");


    @property
    def guide_image(self):
        if(self._guide_image is None):
            if(os.path.exists(self.guide_image_path)):
                self._guide_image = GuideImage(path=os.path.join(self.guide_image_path));
        return self._guide_image;

    @guide_image.setter
    def guide_image(self, value):
        self._guide_image = value;
        if(self._guide_image is not None):
            self._guide_image.writeToFile(self.guide_image_path);
            self._guide_image = GuideImage(path=self.guide_image_path);

    def getGuideImage(self, size):
        return self.guide_image.GetScaled(size).GetRGBCopy().GetFloatCopy();
    

    @property
    def guide_scale(self):
        return self.guide_image.guide_scale;

    @property
    def guide_image_scaled(self):
        return self.guide_image.scaled

    # <editor-fold desc="Property: 'sqrt_alpha_cumprods'">
    @property
    def sqrt_alpha_cumprods(self):
        return self.getInfo("sqrt_alpha_cumprods");
    @sqrt_alpha_cumprods.setter
    def sqrt_alpha_cumprods(self, value):
        self.setInfo('sqrt_alpha_cumprods', value);
    # </editor-fold>



    # def getPathForParams(self, **kwargs):
    #     result_file_name = "";
    #     for key in kwargs:
    #         result_file_name = result_file_name+"_{}{}".format(key, kwargs[key]);
    #     result_file_name = result_file_name+'.png';
    #     return os.path.join(self.getDirectory('results', result_file_name));
    #
    # def loadResultFromPath(self, path=None):
    #     return Image(path=path);

    def Evaluate(self, **kwargs):
        raise NotImplementedError;


    def showdStructureValues(self):
        ts =np.array(self.timesteps);
        plt.figure(figsize=[10,5])
        plt.subplot(211)
        plot = plt.plot(
            ts[:len(self.structure_residuals)], self.structure_residuals
        );
        plt.xlim(ts.max(), ts.min())
        plt.subplot(212)
        plt.plot(ts[:len(self.weight_schedule)], self.weight_schedule);
        plt.xlim(ts.max(), ts.min())

    def initDirs(self, **kwargs):
        super(FGDExperiment, self).initDirs(**kwargs);
        # self.addDirIfMissing(name='results', folder_name="results");
        # self.addDirIfMissing(name='vis', folder_name="vis");

    def adaptiveFilter(self, mean_t, t, sqrt_alpha_cumprod, noise=None, **kwargs):
        # This is where the magic happens
        # def adaptiveFilter(self, sample, mean, noise, t, sqrt_alpha_cumprod, log_variance, **kwargs):
        '''
        sample, mean, noise, t, sqrt_alpha_cumprod, and log_variance come from p_sample in gaussian_diffusion.
        Most of these aren't used. In fact, only sample, mean, and sqrt_alpha_cumprod are really essential, though I also use t for convenience.
        '''
        self.timesteps.append(t);
        self.sqrt_alpha_cumprods.append(sqrt_alpha_cumprod);
        self.means.append(mean_t);

        #####
        def measureNorm(imin):
            '''
            input is numpy array
            measures the norm of a difference image.
            This is just L1/(number of dimensions)
            '''
            return np.linalg.norm(imin.ravel(),ord=1)/(np.prod(imin.shape));

        def modStructure(mean):
            '''
            modifies the structure a bit in each iteration
            '''

            #mean and sample as images
            meanim = tensor2img(mean);
            # sampim = tensor2img(samp);

            # filtered versions of each; self.guide_contrast is generally set to 1 right now
            # it might make sense to examine more whether multiplying guidefiltered by sqrt_alph_cumprod makes the most sense
            guidefiltered = self.guide_image.getFiltered(self.sigmas)*self.guide_contrast*sqrt_alpha_cumprod
            meanfiltered = self.guide_image.applyFilterTo(meanim, self.sigmas);

            # this is ugly, but I'm getting the noise by subtracting the mean from the sample

            # if first round, set replacement strength to 0
            if(len(self.d_structure_values)<1):
                weight = 0;
                self.weight_schedule.append(0);
            else:
                # look at the previous error
                lastError = self.d_structure_values[-1];

                # control weight scales as structure residual gets close to self.target_dstructure
                # this naming isn't great
                weight_control = np.clip(lastError/self.target_dstructure, 0, 1);

                # optionally raise to some exponent. exponent>1 will make control more lenient
                weight_control = weight_control**self.target_control_exponent;

                # weigh the control by the estimated signal strength
                weight = weight_control*sqrt_alpha_cumprod;

                # add weight to record of schedule
                self.weight_schedule.append(weight_control);

            # calculate the difference in structure
            d_structure = guidefiltered-meanfiltered;

            # add diff in structure
            self.d_structures.append(d_structure);

            # measure structure residual
            d_structure_value=measureNorm(d_structure.fpixels);
            if(self.guide_image._mask is not None):
                maskpix = np.clip((
                        1-(1-self.guide_image._mask.fpixels)*self.mask_strength)
                    , 0,1
                );
                d_structure.pixels = d_structure.pixels*np.dstack([maskpix,maskpix,maskpix]);

            # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
            self.d_structure_values.append(d_structure_value);

            # new mean is mean plus weight times the change in structure
            newmean = meanim+weight*(d_structure);

            # add the original noise back to the new mean
            return img2tensor(newmean);



        # if outside of range where we are mixing, still calculate the d_structure but just return the original sample,
        if(t>self.t_start or t<self.t_end):
            return modStructure(mean_t).to(mean_t.device)+noise;
        else:
            dummy = modStructure(mean_t).to(mean_t.device)+noise;
            return mean_t+noise;


    def p_sample(self, sample, mean, noise, t, sqrt_alpha_cumprod, log_variance, **kwargs):
        return sample;


