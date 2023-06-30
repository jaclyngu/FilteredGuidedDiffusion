from apy.amedia import *
import acimops
import torch

class GuideImage(Image):
    def __init__(self, path = None, pixels = None, convert_to_float=True, mask = None, hdr = False, **kwargs):
        super(GuideImage, self).__init__(path = path, pixels = pixels, convert_to_float=convert_to_float, **kwargs);
        self.guide_scale = [64,64];
        im = Image(pixels = self.pixels);
        #self._scaled = im.GetScaled(self.guide_scale).GetFloatCopy();
        self.jbm = None;
        self.jbmsigmas = None;
        self.latent_structure = None;
        self.latent_im = None;
        self.latent_dim = None;
        self._mask = None;
        if(im.n_color_channels>3):
            alpha_mask = im.fpixels[:,:,3];
            im.pixels = im.pixels[:,:,:3];
            maskpix = Image(pixels=alpha_mask);
            self._mask = maskpix.GetScaled(self.guide_scale);

        self._scaled = im.GetScaled(self.guide_scale).GetFloatCopy();
        self._scaled.pixels = (self._scaled.pixels*2)-1;

        if(mask is not None):
            mpix = mask;
            if(isinstance(mask, Image)):
                mpix = mask.fpixels;
            self._mask = Image(pixels=mpix).GetScaled(self.guide_scale).GetGrayCopy().GetFloatCopy();
            
        self._bilateral_matrices = [];

    @property
    def scaled(self):
        return self._scaled;

    # <editor-fold desc="Property: 'guide_scale'">
    @property
    def guide_scale(self):
        return self.getInfo("guide_scale");
    @guide_scale.setter
    def guide_scale(self, value):
        self.setInfo('guide_scale', value);
    # </editor-fold>

    def getCrossBilateralMatrix(self, sigmas):
        for bm in self._bilateral_matrices:
            if(bm[0]==sigmas):
                return bm[1];
        bmat = acimops.bilateral.getCrossBilateralMatrix(self._scaled.fpixels, sigmas);
        filtered = self._applyFilterTo(self.scaled, bmat);
        rval = [
            sigmas,
            bmat,
            filtered
        ];
        self._bilateral_matrices.append(rval);
        return rval[1];

    def getGaussianCirculantMatrix(self, sigmas):
        for bm in self._bilateral_matrices:
            if(bm[0]==sigmas):
                return bm[1];
        bmat = acimops.bilateral.getGaussianCirculantMatrix(self._scaled.fpixels, sigmas);
        filtered = self._applyFilterTo(self.scaled, bmat);
        rval = [
            sigmas,
            bmat,
            filtered
        ];
        self._bilateral_matrices.append(rval);
        return rval[1];
    
    def getJointBilateralTensor(self, sigmas, channels, device):
        bm = self.getCrossBilateralMatrix(sigmas);
        return torch.Tensor(bm).unsqueeze(0).repeat((channels,1,1)).to(device);

    def setJBM(self, tensor, sigmas):
        self.jbm = tensor;
        self.jbmsigmas = sigmas;


    def getFiltered(self, sigmas):
        for bm in self._bilateral_matrices:
            if(bm[0]==sigmas):
                return bm[2];
        self.getCrossBilateralMatrix(sigmas);
        return self.getFiltered(sigmas);

    def applyFilterTo(self, image, sigmas):
        bmat = self.getCrossBilateralMatrix(sigmas);
        return self._applyFilterTo(image, bmat);

    def _applyFilterTo(self, image, bmat):
        matdim = self.guide_scale[0]*self.guide_scale[1];
        if(image.n_color_channels>1):
            rpix = np.reshape(image.fpixels[:,:,0], matdim);
            gpix = np.reshape(image.fpixels[:,:,1], matdim);
            bpix = np.reshape(image.fpixels[:,:,2], matdim);
            rpix_filtered = bmat@rpix;
            gpix_filtered = bmat@gpix;
            bpix_filtered = bmat@bpix;
            return Image(pixels=np.dstack([
                np.reshape(rpix_filtered, self.guide_scale),
                np.reshape(gpix_filtered, self.guide_scale),
                np.reshape(bpix_filtered, self.guide_scale)]
            ));
        else:
            rpix = np.reshape(image.fpixels, matdim);
            rpix_filtered = bmat@rpix;
            return Image(pixels= np.reshape(rpix_filtered, self.guide_scale));





