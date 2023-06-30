# Gaussian convolution applied at each pixel in an image (Cython version)
# Parameters:
#   im: Image to filter, provided as a numpy array
#   sigma:  The variance of the Gaussian (e.g., 1)
#

# Tell Cython to use the faster C libraries.
# If we add any additional math functions, we could also add their C versions here.
from libc.math cimport exp
from libc.math cimport sqrt

import numpy as np

cdef double EPSILON = 1e-5;

# ################################################################
# This function shows how to evaluate a Gaussian with 0 mean and standard deviation sigma when sigma=0 corresponds to a dirac delta.
def nGauss(double x, double sigma):
    # Special case if sigma is 0. In this case we should have a dirac delta.
    if(sigma<EPSILON):
        if(x<EPSILON):
            return 1.0;
        else:
            return 0;
    return exp(-(x**2)/(2 * sigma**2));
# ################################################################################


def gaussianBlur(double[:, :, :] im, double sigma):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #

    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :, :] img_filtered = np.zeros([height, width, 3])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    cdef int n = np.int(sqrt(sigma) * 3)

    cdef int p_y, p_x, i, j, q_y, q_x

    cdef double g, gpr, gpg, gpb
    cdef double w = 0

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            for i in range(-n, n):
                for j in range(-n, n):
                    q_y = max([0, min([height - 1, p_y + i])])
                    q_x = max([0, min([width - 1, p_x + j])])
                    g = exp( -(((q_x - p_x)**2 + (q_y - p_y)**2)) / (2 * sigma**2) )

                    gpr += g * im[q_y, q_x, 0]
                    gpg += g * im[q_y, q_x, 1]
                    gpb += g * im[q_y, q_x, 2]
                    w += g

            img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
            img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
            img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)

    return img_filtered




def gaussianBlurAsymmetric(double[:, :, :] im, double[:] sigma):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #

    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :, :] img_filtered = np.zeros([height, width, 3])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    # cdef int ny = np.int(sqrt(sigma[0]) * 3);
    # cdef int nx = np.int(sqrt(sigma[1]) * 3);

    cdef int ny = min(np.int(sigma[0] * 6),height);
    cdef int nx = min(np.int(sigma[1] * 6), width);

    if(ny<1):
        ny=1;
    if(nx<1):
        nx=1;


    cdef int p_y, p_x, i, j, q_y, q_x

    cdef double g, gpr, gpg, gpb
    cdef double w = 0

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            for i in range(-ny, ny):
                for j in range(-nx, nx):
                    q_y = max([0, min([height - 1, p_y + i])])
                    q_x = max([0, min([width - 1, p_x + j])])
                    g = nGauss(q_x - p_x,sigma[1])*nGauss(q_y - p_y,sigma[0]);
                    # g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )

                    gpr += g * im[q_y, q_x, 0]
                    gpg += g * im[q_y, q_x, 1]
                    gpb += g * im[q_y, q_x, 2]
                    w += g

            img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
            img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
            img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)

    return img_filtered

# def nGauss5DBilateral(double[:] x, double[:] val, double[:] sigma):
#     # Special case if sigma is 0. In this case we should have a dirac delta.
#     for()
#     if(sigma<EPSILON):
#         if(x<EPSILON):
#             return 1.0;
#         else:
#             return 0;
#     return exp(-(x**2)/(2 * sigma**2));

def nGaussExp(double x, double sigma):
    # Special case if sigma is 0. In this case we should have a dirac delta.
    if(sigma<EPSILON):
        if(x<EPSILON):
            return 1.0;
        else:
            return 0;
    return -(x**2)/(2 * sigma**2);

def bilateralNaive(double[:, :, :] im, double[:] sigma, int nx, int ny):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #

    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :, :] img_filtered = np.zeros([height, width, 3])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    # cdef int ny = np.int(sqrt(sigma[0]) * 3);
    # cdef int nx = np.int(sqrt(sigma[1]) * 3);
    # cdef int ny = min(np.int(sigma[0] * 2.5),height);
    # cdef int nx = min(np.int(sigma[1] * 2.5), width);

    if(ny<1):
        ny=1;
    if(nx<1):
        nx=1;


    cdef int p_y, p_x, i, j, q_y, q_x

    cdef double g, ge, gpr, gpg, gpb;
    cdef double[:] pval;
    cdef double w = 0;

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            pval = im[p_y,p_x,:];
            for i in range(-ny, ny):
                for j in range(-nx, nx):
                    q_y = max([0, min([height - 1, p_y + i])])
                    q_x = max([0, min([width - 1, p_x + j])])

                    ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,1]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,2]-pval[2], sigma[2]);
                    # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0]);
                    g=exp(ge);
                    # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[2], sigma[2]);
                    # g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )

                    gpr += g * im[q_y, q_x, 0]
                    gpg += g * im[q_y, q_x, 1]
                    gpb += g * im[q_y, q_x, 2]
                    w += g

            img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
            img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
            img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)

    return img_filtered

def singlePixelMask(double[:, :, :] im, long [:] pixel_coordinates, double[:] pixel_value, double sigma_space, double sigma_value):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #
    cdef int nvals = len(pixel_value);
    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :] mask = np.zeros([height, width])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    # cdef int n = np.int(sqrt(sigma_space)*3);
    cdef int n = np.int(sigma_space)*3;

    if(n<1):
        n=1;

    cdef int p_y, p_x
    cdef double g, ge, gpr, gpg, gpb;
    cdef double[:] pval;
    cdef double w = 0;

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels

    for i in range(-n, n):
        for j in range(-n, n):
            p_y = max([0, min([height - 1, pixel_coordinates[0] + i])])
            p_x = max([0, min([width - 1, pixel_coordinates[1] + j])])

            ge=nGaussExp(p_x - pixel_coordinates[1],sigma_space)+nGaussExp(p_y - pixel_coordinates[0],sigma_space);
            for v in range(nvals):
                ge = ge+nGaussExp(im[p_y,p_x,v] - pixel_value[v],sigma_value);
            g=exp(ge);
            mask[p_y,p_x]=g;
    return mask;

def crossBilateral(double[:, :, :] im, double[:,:,:] mask, double[:] sigma, int nx, int ny):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #

    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :, :] img_filtered = np.zeros([height, width, 3])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    # cdef int ny = np.int(sqrt(sigma[0]) * 3);
    # cdef int nx = np.int(sqrt(sigma[1]) * 3);
    # cdef int ny = height;
    # cdef int nx = width;
    # cdef int ny = min(np.int(sigma[0] * 6),height);
    # cdef int nx = min(np.int(sigma[1] * 6), width);

    if(ny<1):
        ny=1;
    if(nx<1):
        nx=1;


    cdef int p_y, p_x, i, j, q_y, q_x

    cdef double g, ge, gpr, gpg, gpb;
    cdef double[:] pval;
    cdef double w = 0;

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            pval = mask[p_y,p_x,:];
            for i in range(-ny, ny):
                for j in range(-nx, nx):
                    q_y = max([0, min([height - 1, p_y + i])])
                    q_x = max([0, min([width - 1, p_x + j])])

                    ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(mask[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(mask[q_y,q_x,1]-pval[1], sigma[2])+nGaussExp(mask[q_y,q_x,2]-pval[2], sigma[2]);
                    # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0]);
                    g=exp(ge);
                    # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[2], sigma[2]);
                    # g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )

                    gpr += g * im[q_y, q_x, 0]
                    gpg += g * im[q_y, q_x, 1]
                    gpb += g * im[q_y, q_x, 2]
                    w += g

            img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
            img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
            img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)

    return img_filtered


def crossBilateralMatrix(double[:, :, :] im, double[:] sigma):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #
    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :] cbMat = np.zeros([width*height, width*height])
    cdef int p_y, p_x, i, j, q_y, q_x, row_counter, column_counter
    cdef double g, ge, gpr, gpg, gpb;
    cdef double[:] pval;
    cdef double w = 0;

    row_counter = 0;

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            pval = im[p_y,p_x,:];
            column_counter=0;
            for q_y in range(height):
                for q_x in range(width):
                    ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,1]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,2]-pval[2], sigma[2]);
                    g=exp(ge);
                    cbMat[row_counter,column_counter]=g;
                    w += g
                    column_counter=column_counter+1;

            # cbMat[row_counter]=cbMat[row_counter]*(1.0/w);
            for column_counter in range(width*height):
                cbMat[row_counter, column_counter]=cbMat[row_counter, column_counter]*(1.0/w);

            row_counter=row_counter+1;

    return cbMat;


def gaussianBlurCirculantMatrix(double[:, :, :] im, double[:] sigma):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   #
    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :] cbMat = np.zeros([width*height, width*height])
    cdef int p_y, p_x, i, j, q_y, q_x, row_counter, column_counter
    cdef double g, ge, gpr, gpg, gpb;
    cdef double[:] pval;
    cdef double w = 0;

    row_counter = 0;

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            pval = im[p_y,p_x,:];
            column_counter=0;
            for q_y in range(height):
                for q_x in range(width):
                    ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0]);
                    g=exp(ge);
                    cbMat[row_counter,column_counter]=g;
                    w += g
                    column_counter=column_counter+1;

            # cbMat[row_counter]=cbMat[row_counter]*(1.0/w);
            for column_counter in range(width*height):
                cbMat[row_counter, column_counter]=cbMat[row_counter, column_counter]*(1.0/w);

            row_counter=row_counter+1;

    return cbMat;





##################//--Bilateral Grid--\\##################
# <editor-fold desc="Bilateral Grid">
#
# def bilateralGrid(double[:, :, :] im, double[:,:,:] guide, double sigma_space, double sigma_val):
#     cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
#     cdef int width = im.shape[1]   #
#     cdef nvals = im.shape[2];
#     cdef double[:,:,:,:] grid = createGrid(im, guide, sigma_space, sigma_val);
#
#
#
# # def createGrid(double[:,:] im, double[:,:] guide, double sigma_space, double sigma_value):
# #     cdef int o_height = im.shape[0];
# #     cdef int o_width = im.shape[1];
# #     cdef int height = int(np.ceil(np.true_divide(o_height, sigma_space));
# #     cdef int width = int(np.ceil(np.true_divide(o_width, sigma_space));
# #     cdef int nvals = int(np.ceil(np.true_divide(1.0, sigma_value));
# #     cdef double[:, :, :,:] grid = np.zeros([height, width, nvals,2]);
# #     cdef int gy, gx,vi;
# #     for py in range(o_height):
# #         for px in range(o_width):
# #             gy = int(np.round(py/sigma_space));
# #             gx = int(np.round(px/sigma_space));
# #             vi=int(np.round(guide[py,px]/sigma_value));
# #             grid[gy,gx,vi,0]+=im[py,px];
# #             grid[gy,gx,vi,1]+=1.0;
# #     return grid;
#
# def gridBlur1D(double[:, :, :, :] grid, int dim):
#     cdef int height = grid.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
#     cdef int width = grid.shape[1]   #
#     cdef nvals = grid.shape[2];
#
#     # cdef double[:, :, :] to store this as a 3D array of doubles
#     cdef double[:, :, :] grid_filtered = np.zeros(grid.shape);
#     n=1;
#     cdef int p_y, p_x, i, v, q_y, q_x
#     cdef double g
#     cdef double[:] gvals;
#     cdef double[:] avals;
#     cdef double w = 0;
#     # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
#     for p_y in range(height):
#         for p_x in range(width):
#             w = 0
#             gvals= np.zeros([nvals]);
#             avals= np.zeros([nvals]);
#
#             for i in range(-n, n):
#                 if(dim==0):
#                     q_y = max([0, min([height - 1, p_y + i])])
#                     q_x = p_x;
#                 elif(dim==1):
#                     q_x = max([0, min([width - 1, p_x + i])])
#                     q_y = p_y;
#                     # g = exp( -(((q_x - p_x)**2 + (q_y - p_y)**2)) / (2 * sigma**2) )
#                 g = exp( -(((q_x - p_x)**2 + (q_y - p_y)**2))*0.5);
#                 for v in range(nvals):
#                     gvals[v]+=g*grid[q_y,q_x,v,0];
#                     avals[v]+=g*grid[q_y,q_x,v,1];
#                 # w += g;
#             for v in range(nvals):
#                 grid_filtered[p_y,p_x,v,0]=gvals[v];
#                 grid_filtered[p_y,p_x,v,1]=avals[v];
#     return grid_filtered;
#
# def gridBlurSep(double[:, :, :] im, double sigma):
#     cdef int height = im.shape[0];
#     cdef int width = im.shape[1];
#     cdef double[:, :, :] img_filteredY = gridBlur1D(im, sigma, 0);
#     cdef double[:, :, :] img_filteredXY = gridBlur1D(img_filteredY, sigma, 1);
#     return img_filteredXY;
#
# def blurBilateralGrid(double[:,:] im, double sigma_space, double sigma_value):
#     cdef int o_height = im.shape[0];
#     cdef int o_width = im.shape[1];
#     cdef int height = int(np.ceil(np.true_divide(o_height, sigma_space));
#     cdef int width = int(np.ceil(np.true_divide(o_width, sigma_space));
#     cdef int nvals = int(np.ceil(np.true_divide(1.0, sigma_value));
#
#
#     cdef double[:, :, :] grid = np.zeros([height, width, nvals]);
#     cdef double[:, :, :] alphas = np.zeros([height, width, nvals]);
#
#     cdef int p_y, p_x, i, j, q_y, q_x
#     cdef double g, ge, gpr, gpg, gpb;
#     cdef double[:] pval;
#     cdef double w = 0;
#
#     # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
#     for p_y in range(o_height):
#         for p_x in range(o_width):
#             gpr = 0
#             gpg = 0
#             gpb = 0
#             w = 0
#             pval = strokes[p_y,p_x,:];
#             for i in range(-n_space, n_space):
#                 for j in range(-n_space, n_space):
#                     q_y = max([0, min([height - 1, p_y + i])])
#                     q_x = max([0, min([width - 1, p_x + j])])
#
#                     ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(mask[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(mask[q_y,q_x,1]-pval[1], sigma[2])+nGaussExp(mask[q_y,q_x,2]-pval[2], sigma[2]);
#                     # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0]);
#                     g=exp(ge);
#                     # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[2], sigma[2]);
#                     # g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )
#
#                     gpr += g * im[q_y, q_x, 0]
#                     gpg += g * im[q_y, q_x, 1]
#                     gpb += g * im[q_y, q_x, 2]
#                     w += g
#
#             img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
#             img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
#             img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)
#
#     return img_filtered
#
#
#
#
# def filterStrokes(double[:, :, :] strokes, double[:,:,:] image, double sigma_space, double sigma_val):
#     cdef int s_height = strokes.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
#     cdef int s_width = strokes.shape[1]   #
#
#     cdef int i_height = image.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
#     cdef int i_width = image.shape[1]   #
#
#     # cdef double[:, :, :] to store this as a 3D array of doubles
#     cdef double[:, :, :] strokes_filtered = np.zeros([s_height, s_width, 3])
#
#     # A Gaussian has infinite support, but most of it's mass lies within
#     # three standard deviations of the mean. The standard deviation is
#     # the square of the variance, sigma.
#     cdef int n_space = 3*sigma_space;
#
#     if(n_space<1):
#         n_space=1;
#
#     cdef int p_y, p_x, i, j, q_y, q_x
#     cdef double g, ge, gpr, gpg, gpb;
#     cdef double[:] pval;
#     cdef double w = 0;
#
#     # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
#     for p_y in range(i_height):
#         for p_x in range(i_width):
#             gpr = 0
#             gpg = 0
#             gpb = 0
#             w = 0
#             pval = strokes[p_y,p_x,:];
#             for i in range(-n_space, n_space):
#                 for j in range(-n_space, n_space):
#                     q_y = max([0, min([height - 1, p_y + i])])
#                     q_x = max([0, min([width - 1, p_x + j])])
#
#                     ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(mask[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(mask[q_y,q_x,1]-pval[1], sigma[2])+nGaussExp(mask[q_y,q_x,2]-pval[2], sigma[2]);
#                     # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0]);
#                     g=exp(ge);
#                     # ge = nGaussExp(q_x - p_x,sigma[1])+nGaussExp(q_y - p_y,sigma[0])+nGaussExp(im[q_y,q_x,0]-pval[0], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[1], sigma[2])+nGaussExp(im[q_y,q_x,0]-pval[2], sigma[2]);
#                     # g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )
#
#                     gpr += g * im[q_y, q_x, 0]
#                     gpg += g * im[q_y, q_x, 1]
#                     gpb += g * im[q_y, q_x, 2]
#                     w += g
#
#             img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
#             img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
#             img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)
#
#     return img_filtered
#
#
#



# </editor-fold>
##################\\--Bilateral Grid--//##################

