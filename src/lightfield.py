import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.interpolate import RegularGridInterpolator
import cv2


image = plt.imread('../data/chessboard_lightfield.png')
shapeOriginal = image.shape

height = int(shapeOriginal[0] / 16)
width = int(shapeOriginal[1] / 16)
image = image.reshape((height, 16, width, 16, 3))
image = np.moveaxis(image, [0, 2], [2, 3])
# u = 16, v = 16, s = height, t = width


## Sub-aperture views
mosaic = np.zeros(shapeOriginal)
for i in range(16):
    for j in range(16):
        mosaic[i*height:(i+1)*height, j*width:(j+1)*width] = image[i, j]
plt.imsave('mosaic.png', mosaic)


# Refocusing and aperture
lensletSize = 16
maxUV = (lensletSize - 1) / 2
u = np.arange(lensletSize) - maxUV
v = np.flip(np.arange(lensletSize) - maxUV)

refocused = np.zeros((height, width, 3))
d = 1.6

range = range(0, 16)
for i in range:
    for j in range:
        shift = RegularGridInterpolator((np.arange(height), np.arange(width)), image[i, j], method='linear', bounds_error=False, fill_value=None)
        jj, ii = np.meshgrid(np.arange(width), np.arange(height))
        ii = ii.astype(np.float64)
        jj = jj.astype(np.float64)
        ii += d * u[i]
        jj += d * v[j]
        shifted = shift((ii, jj))
        refocused += shifted
refocused /= (len(range) ** 2)
plt.imshow(refocused)
plt.show()
plt.imsave('refocus_a16_f1.6.png', refocused)


## All-in-focus and depth from focus
# We will load the stack of 5 images derived from previous step to save running time here
focalStack = np.zeros((5, height, width, 3))
focalStack[0] = plt.imread('refocus_a16_f0.0.png')[:, :, 0:3]
focalStack[1] = plt.imread('refocus_a16_f0.4.png')[:, :, 0:3]
focalStack[2] = plt.imread('refocus_a16_f0.8.png')[:, :, 0:3]
focalStack[3] = plt.imread('refocus_a16_f1.2.png')[:, :, 0:3]
focalStack[4] = plt.imread('refocus_a16_f1.6.png')[:, :, 0:3]

sigma1 = 3
sigma2 = 5
luminance = np.zeros((5, height, width))
for i in range(5) :
    luminance[i] = skimage.color.rgb2xyz(focalStack[i])[:, :, 1]
highFreq = np.zeros(luminance.shape)
for i in range(5) :
    highFreq[i] = cv2.GaussianBlur(luminance[i], (-1, -1), sigma1)
    highFreq[i] = luminance[i] - highFreq[i]
    highFreq[i] = highFreq[i] ** 2
weight = np.zeros(luminance.shape)
for i in range(5) :
    weight[i] = cv2.GaussianBlur(highFreq[i], (-1, -1), sigma2)
allInFocus = np.zeros(focalStack[0].shape)
depth = np.zeros(weight[0].shape)
for i in range(5) :
    allInFocus += np.moveaxis(np.stack((weight[i], weight[i], weight[i])), 0, -1) * focalStack[i]
    depth += (i * 0.4) * weight[i]
weightSum = np.sum(weight, axis=0)
allInFocus /= np.moveaxis(np.stack((weightSum, weightSum, weightSum)), 0, -1)
depth /= weightSum
depth /= np.max(depth)
plt.imshow(allInFocus)
plt.show()
plt.imshow(depth, cmap='gray')
plt.show()
plt.imsave('allInFocus.png', allInFocus)
plt.imsave('depthFromFocus.png', depth, cmap='gray')


## Confocal stereo
# Again, we will load the image stack from saved images
# (aperture, focus, height, weight, channel)
focalApertureStack = np.zeros((4, 5, height, width, 3))
collage = np.zeros((4*height, 5*width, 3))
for i in range(focalApertureStack.shape[0]) :
    for j in range(focalApertureStack.shape[1]) :
        filename = 'refocus_a' + str(4*(i+1)) + '_f' + f"{(j*0.4):.1f}" + '.png'
        focalApertureStack[i, j] = plt.imread(filename)[:, :, 0:3]
        collage[i*height:(i+1)*height, j*width:(j+1)*width, :] = focalApertureStack[i, j]
variance = np.var(focalApertureStack, axis=0)
variance = np.sum(variance, axis=-1)
depth = np.argmin(variance, axis=0).astype(np.float64)
print(depth.shape)
depth /= np.max(depth)
plt.imshow(collage)
plt.show()
plt.imshow(depth, cmap='gray')
plt.show()
plt.imsave('collage.png', collage)
plt.imsave('confocalStereo.png', depth, cmap='gray')