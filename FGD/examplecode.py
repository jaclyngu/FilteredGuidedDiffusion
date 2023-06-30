import numpy as np

def adaptiveFilter(self, sample, mean, noise, t, sqrt_alpha_cumprod, log_variance, **kwargs):
    '''
    sample, mean, noise, t, sqrt_alpha_cumprod, and log_variance come from p_sample in gaussian_diffusion.
    Most of these aren't used. In fact, only sample, mean, and sqrt_alpha_cumprod are really essential, though I also use t for convenience.
    '''
    def measureNorm(imin):
        '''
        This is just L1/(number of dimensions)
        '''
        return np.linalg.norm(imin.ravel(),ord=1)/(np.prod(imin.shape));

    ##################################Most of the work happens in this subroutine#######################################
    def modStructure(mean, samp):
        '''
        modifies the structure a bit in each iteration
        '''
        #mean and sample as images
        meanim = tensor2img(mean);
        noiseim = temsor2img(noise);

        # filtered versions of each;
        # The joint bilateral matrix is calculated and stored upon the first time this is called, so for most iterations
        # it has already been computed
        # Note that the version that does this as a pytorch tensor multiply is faster, but less clear to read, so we are
        # including this one
        guidefiltered = self.guide_image.getFiltered(self.sigmas)*sqrt_alpha_cumprod
        meanfiltered = self.guide_image.applyFilterTo(meanim, self.sigmas);

        # if first round, set replacement strength to 0
        if(len(self.structure_residuals)<1):
            weight = 0;
            self.weight_schedule.append(0);
        else:
            # look at the previous error
            lastError = self.structure_residuals[-1];

            # control weight scales as structure residual gets close to self.detail_parameter
            # this naming isn't great
            weight_control = np.clip(lastError/self.detail_parameter, 0, 1);

            # weigh the control by the estimated signal strength
            weight = weight_control*sqrt_alpha_cumprod;

        # calculate the difference in structure
        d_structure = guidefiltered-meanfiltered;

        # measure structure residual
        d_structure_value=measureNorm(d_structure.fpixels);

        # record structure residual for later. like a squirrel stashing away an acord. bit it's a structure residual norm.
        self.structure_residuals.append(d_structure_value);

        # new mean is mean plus weight times the change in structure
        newmean = meanim+weight*(d_structure);

        # add the original noise back to the new mean
        return img2tensor(newmean+noiseim);
    ####################################################################################################################

    # if outside of range where we are mixing, still calculate the d_structure but just return the original sample,
    if(t>self.t_start or t<self.t_end):
        s0 = modStructure(mean[0], sample[0]).to(sample.device);
        return sample;

    # getting here means we are in mixing range
    # do the mixing and return
    return modStructure(mean[0], sample[0]).to(sample.device);