from FGDExperiment import *

class FGDBasicExperiment(FGDExperiment):
    def __init__(self, path=None, **kwargs):
        self.samples = [];
        self.means = [];
        self.noises = [];
        self.log_variances = [];
        self.d_structures = [];
        super(FGDBasicExperiment, self).__init__(path=path, **kwargs);

    @classmethod
    def GetParamString(cls,
                       target_dstructure=None,
                       guide_contrast=1.0,
                       text_guidance_scale=3.0,
                       target_control_exponent=2.0,
                       mask_strength=1.0,
                       **kwargs):
        pstring = FGDExperiment.GetParamString(**kwargs);
        return pstring+"tds{}_tgs{}_tce{}_maskStr{}".format(target_dstructure, text_guidance_scale,target_control_exponent, mask_strength)




    # <editor-fold desc="Property: 'weight_schedule'">
    @property
    def weight_schedule(self):
        return self.getInfo("weight_schedule");
    @weight_schedule.setter
    def weight_schedule(self, value):
        self.setInfo('weight_schedule', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'text_guidance_scale'">
    @property
    def text_guidance_scale(self):
        return self.getInfo("text_guidance_scale");
    @text_guidance_scale.setter
    def text_guidance_scale(self, value):
        self.setInfo('text_guidance_scale', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'guide_contrast'">
    @property
    def guide_contrast(self):
        return self.getInfo("guide_contrast");
    @guide_contrast.setter
    def guide_contrast(self, value):
        self.setInfo('guide_contrast', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'target_dstructure'">
    @property
    def target_dstructure(self):
        return self.getInfo("target_dstructure");
    @target_dstructure.setter
    def target_dstructure(self, value):
        self.setInfo('target_dstructure', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'mask_strength'">
    @property
    def mask_strength(self):
        return self.getInfo("mask_strength");
    @mask_strength.setter
    def mask_strength(self, value):
        self.setInfo('mask_strength', value);
    # </editor-fold>


    # <editor-fold desc="Property: 'target_control_exponent'">
    @property
    def target_control_exponent(self):
        return self.getInfo("target_control_exponent");
    @target_control_exponent.setter
    def target_control_exponent(self, value):
        self.setInfo('target_control_exponent', value);
    # </editor-fold>

    def _setFilePath(self, file_path=None, **kwargs):
        if(os.path.isdir(file_path)):
            super(SavesDirectoriesMixin, self)._setFilePath(
                file_path=os.path.join(file_path, "{}.json".format(self.__class__.__name__)), **kwargs
            );
        else:
            super(SavesDirectoriesMixin, self)._setFilePath(file_path=file_path, **kwargs);

    def setParameters(self,
                      target_dstructure = 15,
                      text_guidance_scale=3.0,
                      target_control_exponent=2.0,
                      guide_contrast=1.0,
                      mask_strength = 1.0,
                      **kwargs):
        self.guide_contrast = guide_contrast;
        self.target_dstructure = target_dstructure;
        self.text_guidance_scale = text_guidance_scale;
        self.target_control_exponent = target_control_exponent;
        self.mask_strength = mask_strength;
        super(FGDExperimentTargetDStructure, self).setParameters(**kwargs);



    def Evaluate(self, **kwargs):
        BackwardsSample.RunDiffusion(self,
                                     guidance_scale=self.text_guidance_scale,
                                     noise_img=None,
                                     direction='backward',
                                     structure=1.0);


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



    # This is where the magic happens
    def p_sample(self, sample, mean, noise, t, sqrt_alpha_cumprod, log_variance, **kwargs):
        '''
        sample, mean, noise, t, sqrt_alpha_cumprod, and log_variance come from p_sample in gaussian_diffusion.
        Most of these aren't used. In fact, only sample, mean, and sqrt_alpha_cumprod are really essential, though I also use t for convenience.
        '''

        # This code just stores values to examine them later.
        self.samples.append(sample);
        self.means.append(mean);
        self.noises.append(noise);
        self.timesteps.append(t);
        self.log_variances.append(log_variance);
        self.sqrt_alpha_cumprods.append(sqrt_alpha_cumprod);
        #####


        def measureNorm(imin):
            '''
            input is numpy array
            measures the norm of a difference image.
            This is just L1/(number of dimensions)
            '''
            return np.linalg.norm(imin.ravel(),ord=1)/(np.prod(imin.shape));

        def modStructure(mean, samp):
            '''
            modifies the structure a bit in each iteration
            '''

            #mean and sample as images
            meanim = tensor2img(mean);
            sampim = tensor2img(samp);

            # filtered versions of each; self.guide_contrast is generally set to 1 right now
            # it might make sense to examine more whether multiplying guidefiltered by sqrt_alph_cumprod makes the most sense
            guidefiltered = self.guide_image.getFiltered(self.sigmas)*self.guide_contrast*sqrt_alpha_cumprod
            meanfiltered = self.guide_image.applyFilterTo(meanim, self.sigmas);

            # this is ugly, but I'm getting the noise by subtracting the mean from the sample
            noiseim = sampim-meanim;

            # if first round, set replacement strength to 0
            if(len(self.structure_residuals)<1):
                weight = 0;
                self.weight_schedule.append(0);
            else:
                # look at the previous error
                lastError = self.structure_residuals[-1];

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
            self.structure_residuals.append(d_structure_value);

            # new mean is mean plus weight times the change in structure
            newmean = meanim+weight*(d_structure);

            # add the original noise back to the new mean
            return img2tensor(newmean+noiseim);



        # if outside of range where we are mixing, still calculate the d_structure but just return the original sample,
        if(t>self.t_start or t<self.t_end):
            s0 = modStructure(mean[0], sample[0]).to(sample.device);
            return sample;

        # getting here means we are in mixing range
        # do the mixing and return
        s0 = modStructure(mean[0], sample[0]).to(sample.device);
        s1 = sample[1].unsqueeze(0)
        return th.cat((s0,s1),dim=0);

    def getGuideScaledForComparison(self, size=[256,256]):
        return Image(pixels=self.guide_image.fpixels).GetScaled(size).GetRGBCopy()



    def writeDiffusionVideo(self):
        opath = os.path.join(self.vis_dir, "diffvis_"+self.getParamString()+".mp4");
        vw = VideoWriter(opath, 30);
        for d in self.means:
            mim = tensor2img(d[0]);
            mim.pixels = (mim.pixels+1)/2;
            vw.writeFrame(mim)
        vw.close()
        return Video(opath);

