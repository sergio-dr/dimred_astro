#!/usr/bin/env python3

# %%
import argparse
from ast import Import
import os
from glob import iglob

import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread, imsave

from skimage.measure import block_reduce
from skimage.color import lab2rgb  # TODO:, ydbdr2rgb, ...
from oklab import Oklab
from skimage.util import img_as_uint

try:
    from xisf import XISF
except ImportError:
    XISF = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

np.set_printoptions(precision=4)


DESC = """
Maps multiple bands or channels (given as a bunch of image files, see below for supported formats) 
into a RGB color image. Input data is transformed by a PCA (Principal Components Analysis) 
algorithm to reduce its dimensionality, effectively compressing the information to three 
components: the first one (which has the most variance) is interpreted as lightness data, while 
the second and third components are interpreted as chroma data. This implementation places this 
lightness/chroma data CIELAB color space and then performs its conversion to RGB. 

The chroma data (a, b components) can be rotated and flipped to generate different color mappings.
The --explore N option is useful to explore different color palettes: it generates a NxN mosaic 
with different values for chroma rotation [CHROMA_ROTATION, CHROMA_ROTATION_END) and flipping. 
Every image in the mosaic shows the chroma command line parameters needed to generate it. 

In some cases, --luma-flip may be needed if an inverted image is generated.

The input files are assumed to be in non-linear stage (i.e., previously stretched), or you may try 
the included but simple --strech option. 

Supported file formats:
* Input files: 
  tif (16 bits), png (16 bits), xisf (float 32/64 bits), npz (float 32 bits, 'data' key)
* Output files: 
  tif (16 bits), xisf (float 32/64 bits), npz (float 32 bits, 'data' key)
* Output files for exploratory mode: 
  tif (8 bits), png (8 bits)

Examples:
* Basic usage:
```
  dimred.py *.tif output\pca.tif -cr 30 
```  

* Exploratory mode, with 5x5 chroma rotation values in the range [30º, 120º], downscaling by 8: 
```
  dimred.py *.tif output\pca.tif -e 5 -cr 30 -cr 120
```

Note: this tool requires the xisf package to read/write XISF files, see
https://github.com/sergio-dr/xisf
"""

DEBUG_CMDLINE = None
# DEBUG_CMDLINE = "*.tif pca.tif -e 5".split(" ")

creator_app = "github.com/sergio-dr/dimred_astro"

epilog = f"Project page: {creator_app}, by Sergio Díaz, sergiodiaz.eu"

DOWNSAMPLED_IM_WIDTH = 640  # px
BASE_SCALE_FACTOR = 100

config_defaults = {
    'n_components': 3,
    'luma_scale_factor': BASE_SCALE_FACTOR,
    'chroma_scale_factor': 0.8 * BASE_SCALE_FACTOR,
    'downscale_factor': 8,
    'quantile_min': 0.01,
    'quantile_max': 99.99,
    'chroma_rotation': 0.0,
    'mask_exponent': 4.0,
    'explore': 0,
}


class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=DESC, epilog=epilog, formatter_class=HelpFormatter)
parser.add_argument("input_files", help="Input files specified using wildcards, e.g. '*.tif'")
parser.add_argument("output_file", help="Output filename")
parser.add_argument(
    "-c",
    "--compress",
    action='store_true',
    help="Enables lz4hc+shuffling compression for output files (XISF only)",
)
parser.add_argument(
    "-qm",
    "--quantile-min",
    type=float,
    default=config_defaults['quantile_min'],
    help="Quantile range for chroma scaling, minimum value (optional)",
)
parser.add_argument(
    "-qM",
    "--quantile-max",
    type=float,
    default=config_defaults['quantile_max'],
    help="Quantile range for chroma scaling, maximum value (optional)",
)
parser.add_argument(
    "-cr",
    "--chroma-rotation",
    type=float,
    default=config_defaults['chroma_rotation'],
    help="Angle (in degrees) for specifying chroma components rotation; will be interpreted as initial angle in --explore mode (optional)",
)
parser.add_argument(
    "-ce",
    "--chroma-rotation-end",
    type=float,
    help="End angle (in degrees) for in --explore mode; by default, 360º+CHROMA_ROTATION (optional)",
)
parser.add_argument(
    "-cf",
    "--chroma-flip",
    action='store_true',
    help="Flips chroma plane, specifically flipping the sign of the first chroma component (optional)",
)
parser.add_argument(
    "-lf", "--luma-flip", action='store_true', help="Flips the sign of the lightness channel"
)
parser.add_argument(
    "-cs",
    "--chroma-scale-factor",
    type=float,
    default=config_defaults['chroma_scale_factor'],
    help="Multiplicative factor to scale chroma components (optional)",
)
parser.add_argument(
    "-ok",
    "--oklab",
    action='store_true',
    help="Use oklab perceptual color space by Björn Ottosson instead of CIELAB (optional)",
)
parser.add_argument(
    "-sm",
    "--star-mask",
    action='store_true',
    help="Applies mask to prevent star cores from getting out of gamut (optional)",
)
parser.add_argument(
    "-me",
    "--mask-exponent",
    type=float,
    default=config_defaults['mask_exponent'],
    help="Defines a simple mask based on lightness: mask = 1 - L^mask_exponent (optional)",
)
parser.add_argument(
    "-s",
    "--stretch",
    action='store_true',
    help="Applies a nonlinear stretch to the data (optional)",
)
parser.add_argument(
    "-e",
    "--explore",
    type=int,
    default=config_defaults['explore'],
    help="Exploratory mode: given an integer N, generates a NxN mosaic with the chroma angle range specified by -cr and -ce (optional)",
)


args = parser.parse_args(DEBUG_CMDLINE)
config = {**config_defaults, **vars(args)}
if DEBUG_CMDLINE:
    print(config)


# %%

# __/ Prepare output directory \__________
output_path = os.path.dirname(os.path.join('.', config['output_file']))
os.makedirs(output_path, exist_ok=True)


# __/ Select image read and write method depending on file extension \__________
image_read = {
    '.tif': lambda fname: np.atleast_3d(imread(fname).astype(np.float32) / (2**16 - 1)),
    '.npz': lambda fname: np.atleast_3d(np.load(fname)['data']),
}

image_write = {
    '.tif': lambda fname, d: imsave(fname, img_as_uint(d)),
    #'.png' : lambda fname, d: imsave(fname, img_as_uint(d)),
    # 8-bit! see https://github.com/imageio/imageio/issues/204#issuecomment-271566703
    '.npz': lambda fname, d: np.savez_compressed(fname, data=d),
}

allowed_explore_formats = ['.tif', '.png']

# Add methods for XISF if module is installed
if XISF is None:
    print(
        "You may add XISF format support by installing the package 'xisf': https://github.com/sergio-dr/xisf"
    )
else:
    image_read['.xisf'] = XISF.read
    codec, shuffle = ('lz4hc', True) if config['compress'] else (None, False)
    image_write['.xisf'] = lambda fname, d: XISF.write(
        fname, d.astype(np.float32), creator_app=creator_app, codec=codec, shuffle=shuffle
    )


# Pre-check if input/output file formats are supported
_, in_ext = os.path.splitext(config['input_files'])
if not in_ext in image_read.keys():
    print(f"Input file format {in_ext} not supported.\n")
    exit(1)

_, out_ext = os.path.splitext(config['output_file'])
if config['explore'] > 0 and out_ext not in allowed_explore_formats:
    print(f"Only {allowed_explore_formats} formats are supported in explore mode.\n")
    exit(1)
elif config['explore'] == 0 and not out_ext in image_write.keys():
    print(f"Output file format {out_ext} not supported.\n")
    exit(1)


# Oklab or CIE-LAB to RGB converter function


# Oklab tends to produce a darker image compared with CIE-LAB.
# https://bottosson.github.io/posts/colorpicker/ suggest using Lr instead of L:
def oklr(x):  # in place
    k1, k2 = 0.206, 0.3
    k3 = (1 + k1) / (1 + k2)
    L = x[..., 0]
    x[..., 0] = 0.5 * (k3 * L - k1 + np.sqrt((k3 * L - k1) ** 2 + 4 * k2 * k3 * L))
    return x


# def okL_boost(x):  # in place
#     L = x[..., 0]
#     x[..., 0] = L**(2/3)
#     return x

oklab2rgb = lambda x: Oklab.to_rgb(oklr(x / BASE_SCALE_FACTOR))
lab2rgb_fn = oklab2rgb if config['oklab'] else lab2rgb


# __/ Open input files and generate the stack \__________
path_tpl = os.path.join('.', config['input_files'])
images = []
print("Input images:")
for filename in iglob(path_tpl):
    # Select proper image reading method based on file extension
    im = image_read[in_ext](filename)
    print(filename, im.dtype, im.shape)
    images.append(im)

X = np.concatenate(images, axis=-1)
print("... --> ", X.shape, "\n")
h, w, channels = X.shape

del im, images


# %%


# __/ Preprocessing \__________
def mtf(m, x, r=2):
    return ((1 - m) * x) / ((1 - r * m) * x + (r - 1) * m)


def nonlinear_stretch(data):
    data_min, data_max = np.nanmin(data, axis=(0, 1)), np.nanmax(data, axis=(0, 1))
    data = (data - data_min) / (data_max - data_min)
    data = mtf(np.nanmedian(data, axis=(0, 1)), data, 4)
    return data


# Speed up PCA fitting by downsampling
# Compute downsampling factor:
df = int(w / DOWNSAMPLED_IM_WIDTH)

# Train dataset as a downsample of the original data
# X_train = X[::df, ::df, :]  # sampling by strides
# Downscaling by local median:
X_train = block_reduce(X, (df, df, 1), np.median, np.median(X))

# In explore mode, the PCA will also be applied on the downsampled data
if config['explore']:
    X = X[::df, ::df, :]
    h, w, channels = X_train.shape

# Stretch data if requested
if config['stretch']:
    X = nonlinear_stretch(X)

# Replace NaNs (with 0.0) if present
X = np.nan_to_num(X)


# __/ Apply the PCA transformation \__________

# Flatten the spatial axis
X = X.reshape((-1, channels))
X_train = X_train.reshape((-1, channels))

print(f"Applying PCA on a {X_train.shape} sample...")
pca = PCA(n_components=config['n_components'])
pca.fit(X_train)
embedding = pca.transform(X)
evr = pca.explained_variance_ratio_
print("Explained variance ratio: ", evr, "\n")
print("Components:\n", pca.components_, "\n")

# %%

# __/ Scale the principal components \__________
luma_scaler = MinMaxScaler()

quantile_range = (config['quantile_min'], config['quantile_max'])
chroma_scaler = RobustScaler(quantile_range=quantile_range)

luma = config['luma_scale_factor'] * luma_scaler.fit_transform(embedding[:, [0]])
# TODO si en la primera componente hay coeficientes negativos, sopesar si reemplazarla por:
# luma = np.average(X, axis=-1, weights=np.abs(pca.components_[0,:]), keepdims=True)
# luma = config['luma_scale_factor'] * luma_scaler.fit_transform(luma)

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
    return lum_chr * [-1, 1, 1] + [config['luma_scale_factor'], 0, 0]


def chroma_flipping(lum_chr):
    # just need to flip first chroma component;
    # flipping the second is equivalent to flipping the first and rotating -180º;
    # flipping both and rotating -180º is the same as no flipping.
    return lum_chr * [1, -1, 1]


def chroma_rotation(lum_chr, degs):
    rads = degs * (np.pi / 180)
    cos_a = np.cos(rads)
    sin_a = np.sin(rads)

    ch_mapping = np.array([[1, 0, 0], [0, cos_a, sin_a], [0, -sin_a, cos_a]]).T

    return lum_chr @ ch_mapping


# __/ Apply luma/chroma transformations and write output \__________

if config['luma_flip']:
    lum_chr = luma_flipping(lum_chr)

if config['explore'] == 0:
    print("Generating output image...\n")
    if config['chroma_flip']:
        lum_chr = chroma_flipping(lum_chr)

    # Debug Lab output
    # imsave(config['output_file'].replace(".tif", "_L.tif"), img_as_uint(lum_chr[:,0].reshape((h,w,1))/100.0))
    # imsave(config['output_file'].replace(".tif", "_a.tif"), img_as_uint(lum_chr[:,1].reshape((h,w,1))/200.0 + 0.5))
    # imsave(config['output_file'].replace(".tif", "_b.tif"), img_as_uint(lum_chr[:,2].reshape((h,w,1))/200.0 + 0.5))

    lum_chr = chroma_rotation(lum_chr, config['chroma_rotation'])

    print("Lab channels range (after rotation/flipping):")
    lab_ch_names = "Lab"
    for ch in range(config['n_components']):
        print(f"{lab_ch_names[ch]} : {lum_chr[:, ch].min():.2f} .. {lum_chr[:, ch].max():.2f}")
    print()

    lum_chr = lum_chr.reshape((h, w, 3))

    if config['star_mask']:
        print(f"Applying star mask (1-L^{config['mask_exponent']})...\n")
        mask = 1.0 - np.power(
            lum_chr[..., [0]] / config['luma_scale_factor'], config['mask_exponent']
        )
        lum_chr[..., 1:] *= mask

    rgb = lab2rgb_fn(lum_chr)

    image_write[out_ext](config['output_file'], rgb)

else:
    N = config['explore']
    print(f"Generating {N}x{N} mosaic, ", end="")
    M = N * N
    cols, rows = N, 2 * N

    cr, ce = config['chroma_rotation'], config['chroma_rotation_end']
    hue_i = cr
    hue_f = (cr + 360) if ce is None else ce
    print(f"{hue_i:.1f} to {hue_f:.1f}...")

    ar = w / h
    S = 4
    fig, axs = plt.subplots(rows, cols, figsize=(ar * S * cols, S * rows), facecolor='dimgray')
    axs = axs.ravel()
    for i, degs in enumerate(np.linspace(hue_i, hue_f, M, endpoint=False)):
        print(f"Rotation: {degs:.1f}º")
        lum_chr_degs = chroma_rotation(lum_chr, degs).reshape((h, w, 3))
        rgb_hue = lab2rgb_fn(lum_chr_degs)
        axs[i].imshow(rgb_hue)
        axs[i].set_axis_off()
        axs[i].text(0, 16, f"-cr={degs:.1f}", c='white', fontsize=16)
    print("Chroma flip")
    lum_chr = chroma_flipping(lum_chr)
    for i, degs in enumerate(np.linspace(hue_i, hue_f, M, endpoint=False)):
        print(f"Rotation: {degs:.1f}º")
        lum_chr_degs = chroma_rotation(lum_chr, degs).reshape((h, w, 3))
        rgb_hue = lab2rgb_fn(lum_chr_degs)
        axs[i + M].imshow(rgb_hue)
        axs[i + M].set_axis_off()
        axs[i + M].text(0, 16, f"-cr={degs:.1f} -cf", c='white', fontsize=16)
    fig.tight_layout()
    fig.savefig(config['output_file'])

print("... done.\n")

# %%
