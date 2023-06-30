import cbilateral
import numpy as np

# def getGaussianFiltered(image, sigma):
#     # img = np.array(Image.open('../img/rubiks_cube.png'), dtype=np.float32)
#     im=self.GetFloatCopy();
#     sigmas = sigma;
#
#     if(not isinstance(sigmas, (list, tuple, np.ndarray))):
#         img_filtered = np.asarray(cyimage.gaussianBlur(im.pixels, sigmas))
#     else:
#         if(not isinstance(sigmas, np.ndarray)):
#             sigmas = np.array(sigmas);
#         img_filtered = np.asarray(cyimage.gaussianBlurAsymmetric(im.pixels, sigmas.astype('d')));
#     return self.mobjectClass(pixels=img_filtered);

def _isFloat(im):
    return (im.dtype.kind in 'f');

def _getFloatIm(uintim):
    return uintim.astype(np.float)*np.true_divide(1.0,255.0);

def _getUIntIm(floatim):
    return (floatim * 255).astype(np.uint8);

def getGaussianFilteredSep(image, sigma):
    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image.copy();
    else:
        im=_getFloatIm(image);

    img_filtered = np.asarray(cbilateral.gaussianBlurSep(im, sigma))

    if(isFloat):
        return img_filtered;
    else:
        return _getUIntIm(img_filtered);



    # def getFilteredStrokes(self):
    #     '''
    #     should create a 2 layer grid, one for in region, one for out of region
    #     :return:
    #     :rtype:
    #     '''


def getBilateralFiltered(image, sigma, neighborhood=None):
    '''

    :param image:
    :type image:
    :param sigma: sigmas for filter [sy,sx,sv]
    :type sigma: list or tuple of sigmas
    :return:
    :rtype:
    '''
    sigmas = sigma;
    if(not isinstance(sigmas, (list, tuple, np.ndarray))):
        sigmas = np.array([sigma, sigma, sigma]);
    else:
        if(not isinstance(sigmas, np.ndarray)):
            sigmas = np.array(sigmas);

    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image.copy();
    else:
        im=_getFloatIm(image);

    if(neighborhood is None):
        # neighborhood = [int(np.ceil(5*sigmas[0])), int(np.ceil(5*sigmas[1]))]
        neighborhood = [int(np.ceil(2*sigmas[0])), int(np.ceil(2*sigmas[1]))]

    img_filtered = np.asarray(cbilateral.bilateralNaive(im, sigmas, neighborhood[0], neighborhood[1]))

    if(isFloat):
        return img_filtered;
    else:
        return _getUIntIm(img_filtered);

def getCrossBilateralFiltered(image, guide, sigma, neighborhood=None):
    '''

    :param image:
    :type image:
    :param guide:
    :type guide:
    :param sigma: [sy,sx,sv]
    :type sigma:
    :return:
    :rtype:
    '''

    sigmas = sigma;
    if(not isinstance(sigmas, (list, tuple, np.ndarray))):
        sigmas = np.array([sigma, sigma, sigma]);
    else:
        if(not isinstance(sigmas, np.ndarray)):
            sigmas = np.array(sigmas);

    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image.copy();
        refIm = guide.copy();
    else:
        im=_getFloatIm(image);
        refIm = _getFloatIm(guide);

    if(neighborhood is None):
        neighborhood = [int(np.ceil(5*sigmas[0])), int(np.ceil(5*sigmas[1]))]

    print("using neighborhood {}".format(neighborhood));
    img_filtered = np.asarray(cbilateral.crossBilateral(im, refIm, sigmas, neighborhood[0], neighborhood[1]))

    if(isFloat):
        return img_filtered;
    else:
        return _getUIntIm(img_filtered);

def getCrossBilateralMatrix(image, sigma):
    print('getCrossBilateralMatrix')
    sigmas = sigma;
    if(not isinstance(sigmas, (list, tuple, np.ndarray))):
        sigmas = np.array([sigma, sigma, sigma]);
    else:
        if(not isinstance(sigmas, np.ndarray)):
            sigmas = np.array(sigmas);

    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image.copy();
    else:
        im=_getFloatIm(image);

    return np.asarray(cbilateral.crossBilateralMatrix(im, sigmas));

def getGaussianCirculantMatrix(image, sigma):
    sigmas = sigma;
    if(not isinstance(sigmas, (list, tuple, np.ndarray))):
        sigmas = np.array([sigma, sigma]);
    else:
        if(not isinstance(sigmas, np.ndarray)):
            sigmas = np.array(sigmas);

    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image.copy();
    else:
        im=_getFloatIm(image);

    return np.asarray(cbilateral.gaussianBlurCirculantMatrix(im, sigmas));

def _getAsNDArrray(a):
    ar=a;
    if(not isinstance(ar, np.ndarray)):
        ar = np.array(a);
    return ar;

def getBilateralPixelMask(image, pixel_coordinates, sigma_space, sigma_value, pixel_value=None):
    isFloat = (image.dtype.kind in 'f');
    if(isFloat):
        im=image;
    else:
        im=_getFloatIm(image);

    if(pixel_value is None):
        pixel_value=im[pixel_coordinates[0],pixel_coordinates[1]];

    mask_pixels = np.asarray(cbilateral.singlePixelMask(im, _getAsNDArrray(pixel_coordinates), _getAsNDArrray(pixel_value), sigma_space, sigma_value));
    if(isFloat):
        return mask_pixels;
    else:
        return _getUIntIm(mask_pixels);

def sumOfPixelMasks(image, pixel_coordinates, sigma_space, sigma_value):
    mask = np.zeros([image.shape[0], image.shape[1]]);
    if(not isinstance(sigma_space,(list,tuple,np.ndarray))):
        s_space = np.array([sigma_space]*len(pixel_coordinates));
    if(not isinstance(sigma_value,(list,tuple,np.ndarray))):
        s_val = np.array([sigma_value]*len(pixel_coordinates));
    for p in range(len(pixel_coordinates)):
        mask = mask+getBilateralPixelMask(image, pixel_coordinates[p], s_space[p], s_val[p]);
    return mask;

def maxOfPixelMasks(image, pixel_coordinates, sigma_space, sigma_value):
    mask = np.zeros([image.shape[0], image.shape[1]]);
    if(not isinstance(sigma_space,(list,tuple,np.ndarray))):
        s_space = np.array([sigma_space]*len(pixel_coordinates));
    if(not isinstance(sigma_value,(list,tuple,np.ndarray))):
        s_val = np.array([sigma_value]*len(pixel_coordinates));
    for p in range(len(pixel_coordinates)):
        mask = np.maximum(mask, getBilateralPixelMask(image, pixel_coordinates[p], s_space[p], s_val[p]));
    return mask;


