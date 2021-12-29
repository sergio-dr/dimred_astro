#!/usr/bin/env python3

# %%
import argparse
import os
from glob import iglob 

import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread, imsave
from skimage.transform import rescale 
from skimage.color import lab2rgb # TODO:, ydbdr2rgb, ...
from skimage.util import img_as_uint

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

np.set_printoptions(precision=4)


DESC="""
Maps multiple bands or channels (given as a bunch of monochromatic image TIFF files) into a RGB color image. 
Input data is transformed by a PCA (Principal Components Analysis) algorithm to reduce the dimensionality, 
effectively compressing the information to three components: the first one (which has the most variance) is 
interpreted as luminance data, while the second and third components are interpreted as chrominance data. 
This implementation places this luminance/chrominance data CIELAB color space and then performs its 
conversion to RGB. 

The chrominance data (a, b components) can be rotated and flipped to generate different color mappings. 
The --explore N option is useful to explore different color palettes: it generates a NxN mosaic with different
values for chroma rotation [CHROMA_ROTATION, CHROMA_ROTATION_END) and flipping. Be sure to use it along with 
a high DOWNSCALE_FACTOR to speed computations. Every image in the mosaic shows the chroma command line 
parameters needed to generate it. 

In some cases, --luma-flip may be needed if an inverted image is generated.

The input files are assumed to be in non-linear stage (i.e., previously stretched).

Examples:
* Basic usage:
```
  dimred.py *.tif output\pca.tif -cr 30 
```  

* Exploratory mode, with 5x5 chroma rotation values in the range [30º, 120º], downscaling by 8: 
```
  dimred.py *.tif output\pca.tif -e 5 -df 8 -cr 30 -cr 120
```

"""

DEBUG_CMDLINE = None
#DEBUG_CMDLINE = "*.tif pca.tif -df 8 -e 5".split(" ") 

config_defaults = {
    'n_components': 3,
    'luma_scale_factor': 100,
    'chroma_scale_factor': 90,    
    'downscale_factor': 1,
    'quantile_min': 0.01,
    'quantile_max': 99.99,
    'chroma_rotation': 0.0,
    'mask_exponent': 4.0,
    'explore': 0,
}

parser = argparse.ArgumentParser(description=DESC, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("input_files", 
                    help="Input files specified using wildcards, e.g. '*.tif'")
parser.add_argument("output_file", 
                    help="Output filename") 
parser.add_argument("-df", "--downscale-factor", type=int, default=config_defaults['downscale_factor'],
                    help="Integer downscaling factor to speed up computation, as a preview (optional)")
parser.add_argument("-qm", "--quantile-min", type=float, default=config_defaults['quantile_min'],
                    help="Quantile range for chroma scaling, minimum value (optional)")
parser.add_argument("-qM", "--quantile-max", type=float, default=config_defaults['quantile_max'],
                    help="Quantile range for chroma scaling, maximum value (optional)")
parser.add_argument("-cr", "--chroma-rotation", type=float, default=config_defaults['chroma_rotation'],
                    help="Angle (in degrees) for specifying chroma components rotation; will be interpreted as initial angle in --explore mode (optional)")
parser.add_argument("-ce", "--chroma-rotation-end", type=float, 
                    help="End angle (in degrees) for in --explore mode; by default, 360º+CHROMA_ROTATION (optional)")
parser.add_argument("-cf", "--chroma-flip", action='store_true',
                    help="Flips chrominance plane, specifically flipping the sign of the first chroma component (optional)")
parser.add_argument("-lf", "--luma-flip", action='store_true',
                    help="Flips sign of the luminance channel")
parser.add_argument("-cs", "--chroma-scale-factor", type=float, default=config_defaults['chroma_scale_factor'],
                    help="Multiplicative factor to scale chroma components (optional)")
parser.add_argument("-sm", "--star-mask", action='store_true',
                    help="Applies mask to prevent star cores from getting out of gamut (optional)")
parser.add_argument("-me", "--mask-exponent", type=float, default=config_defaults['mask_exponent'],
                    help="Defines a simple mask based on luminance: mask = 1 - L^mask_exponent (optional)") 
parser.add_argument("-e", "--explore", type=int, default=config_defaults['explore'],
                    help="Exploratory mode: given an integer N, generates a NxN mosaic with the --chroma-angle range specified (optional)")


args = parser.parse_args(DEBUG_CMDLINE)
config = { **config_defaults, **vars(args) }
if DEBUG_CMDLINE:
    print(config)


# %%

# __/ Prepare output directory \__________
output_path = os.path.dirname(os.path.join('.', config['output_file']))
os.makedirs(output_path, exist_ok=True)


# __/ Open input files and generate the stack \__________
path_tpl = os.path.join('.', config['input_files'])
images = []
print("Input images:")
for filename in iglob(path_tpl):
    im = imread(filename)
    print(filename, im.dtype, im.shape)        
    im = im.astype(np.float32) / (2**16 - 1)    
    if config['downscale_factor'] > 1:
        im = rescale(im, 1.0/config['downscale_factor'], anti_aliasing=False)
        # rescale with no aliasing (nearest neighbor) to sample the original image data;
        # downscale_local_mean would change the data so PCA components may flip sign
    images.append( im )
print()

data = np.stack(images, axis=-1)
del im, images

# %%

# __/ Apply the PCA transformation \__________
h, w, channels = data.shape
data = data.reshape((-1, channels))

print("Applying PCA...")
pca = PCA(n_components=config['n_components'])
embedding = pca.fit_transform(data)
del data

evr = pca.explained_variance_ratio_
print("Explained variance ratio: ", evr, "\n")


# %%

# __/ Scale the principal components \__________
luma_scaler = MinMaxScaler()

quantile_range = (config['quantile_min'], config['quantile_max'])
chroma_scaler = RobustScaler(quantile_range=quantile_range) 

luma = config['luma_scale_factor'] * luma_scaler.fit_transform(embedding[:, [0]])
chroma = config['chroma_scale_factor'] * chroma_scaler.fit_transform(embedding[:, [1, 2]])

lum_chr = np.concatenate([luma, chroma], axis=-1)

print("Lab channels range:")
lab_ch_names = "Lab"
for ch in range(config['n_components']):
    print(f"{lab_ch_names[ch]} : {lum_chr[:, ch].min():.2f} .. {lum_chr[:, ch].max():.2f}")
print()


# %%

# __/ Luma/Chroma transformations \__________

def luma_flipping(lum_chr):
    return lum_chr * [-1, 1, 1]

def chroma_flipping(lum_chr):
    # just need to flip first chroma component; 
    # flipping the second is equivalent to flipping the first and rotating -180º;
    # flipping both and rotating -180º is the same as no flipping. 
    return lum_chr * [1, -1, 1] 

def chroma_rotation(lum_chr, degs):
    rads = degs * (np.pi/180)
    cos_a = np.cos(rads)
    sin_a = np.sin(rads)

    ch_mapping = np.array([ 
        [1, 0, 0],
        [0, cos_a, sin_a],
        [0, -sin_a, cos_a]
    ]).T

    return lum_chr @ ch_mapping


# __/ Apply luma/chroma transformations and write output \__________

if config['luma_flip']:
    lum_chr = luma_flipping(lum_chr)

if config['explore'] == 0:
    print("Generating output image...\n")
    if config['chroma_flip']:
        lum_chr = chroma_flipping(lum_chr)

    # Debug CIELab output
    #imsave(config['output_file'].replace(".tif", "_L.tif"), img_as_uint(lum_chr[:,0].reshape((h,w,1))/100.0))
    #imsave(config['output_file'].replace(".tif", "_a.tif"), img_as_uint(lum_chr[:,1].reshape((h,w,1))/200.0 + 0.5))
    #imsave(config['output_file'].replace(".tif", "_b.tif"), img_as_uint(lum_chr[:,2].reshape((h,w,1))/200.0 + 0.5))
    
    lum_chr = chroma_rotation(lum_chr, config['chroma_rotation'])

    print("Lab channels range (after rotation/flipping):")
    lab_ch_names = "Lab"
    for ch in range(config['n_components']):
        print(f"{lab_ch_names[ch]} : {lum_chr[:, ch].min():.2f} .. {lum_chr[:, ch].max():.2f}")
    print()

    lum_chr = lum_chr.reshape((h, w, 3))

    if config['star_mask']:
        print(f"Applying star mask (1-L^{config['mask_exponent']})...\n")
        mask = 1.0 - np.power(lum_chr[..., [0]]/config['luma_scale_factor'], config['mask_exponent'])
        lum_chr[..., 1:] *= mask

    rgb = lab2rgb(lum_chr) 
    imsave(config['output_file'], img_as_uint(rgb))
else:
    N = config['explore']
    print(f"Generating {N}x{N} mosaic, ", end="")    
    M = N*N
    cols, rows = N, 2*N
    hue_i = config['chroma_rotation']
    hue_f = config['chroma_rotation_end'] if config['chroma_rotation_end'] is not None else config['chroma_rotation']+360 
    print(f"{hue_i:.1f} to {hue_f:.1f}...")

    ar = w/h
    S = 4
    fig, axs = plt.subplots(rows, cols, figsize=(ar*S*cols, S*rows), facecolor='dimgray')
    axs = axs.ravel()
    for i, degs in enumerate(np.linspace(hue_i, hue_f, M, endpoint=False)):
        lum_chr_degs = chroma_rotation(lum_chr, degs).reshape((h, w, 3))
        rgb_hue = lab2rgb(lum_chr_degs)
        axs[i].imshow(rgb_hue)
        axs[i].set_axis_off()
        axs[i].text(0, 16, f"-cr={degs:.1f}", c='white', fontsize=16)
    lum_chr = chroma_flipping(lum_chr)  
    for i, degs in enumerate(np.linspace(hue_i, hue_f, M, endpoint=False)):
        lum_chr_degs = chroma_rotation(lum_chr, degs).reshape((h, w, 3))
        rgb_hue = lab2rgb(lum_chr_degs)
        axs[i+M].imshow(rgb_hue)
        axs[i+M].set_axis_off()
        axs[i+M].text(0, 16, f"-cr={degs:.1f} -cf", c='white', fontsize=16)
    fig.tight_layout()
    fig.savefig(config['output_file'])

print("... done.\n")
# %%
