import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, default_fp=ti.f64)
EPSILON = 1e-5

@ti.func
def nGaussExp(x: ti.f64, sigma: ti.f64):
    # Special case if sigma is 0. In this case we should have a dirac delta.
    value = 0.0
    if(sigma<EPSILON):
        if(x<EPSILON):
            value = 1.0
    else:
        value = -(x**2)/(2 * sigma**2)
    return value
        

@ti.kernel
def taichiCrossBilateralMatrix4D(im: ti.types.ndarray(dtype=ti.f64, ndim=3), sigma: ti.math.vec3, cbmat: ti.types.ndarray()):
    height = im.shape[0]
    width = im.shape[1]

    for py in range(height):
        for px in range(width):
            pval = ti.math.vec4(im[py,px,0], im[py,px,1], im[py,px,2], im[py,px,3])
            w = 0.0
            row = py*width+px
            for qy in range(height):
                for qx in range(width):
                    ge = nGaussExp(qx - px,sigma[1])+nGaussExp(qy - py,sigma[0])\
                        +nGaussExp(im[qy,qx,0]-pval[0], sigma[2])+nGaussExp(im[qy,qx,1]-pval[1], sigma[2])\
                        +nGaussExp(im[qy,qx,2]-pval[2], sigma[2])+nGaussExp(im[qy,qx,1]-pval[1], sigma[2])
                    g = ti.exp(ge)
                    col = qy*width+qx
                    cbmat[row,col] = g
                    w += g
             
            for col in range(width*height):
                cbmat[row, col]=cbmat[row, col]*(1.0/w)

def getCrossBilateralMatrix4D(image, sigmas):
    height=image.shape[0]
    width = image.shape[1]

    cbmat = np.zeros([width*height,width*height])
    taichiCrossBilateralMatrix4D(np.ascontiguousarray(image), sigmas, cbmat)
    return cbmat

    

    

