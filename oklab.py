# https://bottosson.github.io/posts/oklab/

import numpy as np


class Oklab:
    RGB2LMS = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    ).T

    LMS_2OKLAB = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    ).T

    OKLAB2LMS_ = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    ).T

    LMS2RGB = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    ).T

    @staticmethod
    def _gammainv(x):  # linear_srgb_from_srgb
        return np.piecewise(
            x, [x >= 0.04045], [lambda x: np.power((x + 0.055) / 1.055, 2.4), lambda x: x / 12.92]
        )

    @staticmethod
    def _gamma(x):  # srgb_from_linear_srgb
        return np.piecewise(
            x,
            [x >= 0.0031308],
            [lambda x: 1.055 * np.power(x, 1.0 / 2.4) - 0.055, lambda x: 12.92 * x],
        )

    @classmethod
    def from_rgb(cls, rgb):
        lms = cls._gammainv(rgb) @ cls.RGB2LMS
        lms_ = np.cbrt(lms)
        return lms_ @ cls.LMS_2OKLAB

    @classmethod
    def to_rgb(cls, oklab):
        lms_ = oklab @ cls.OKLAB2LMS_
        lms = lms_ * lms_ * lms_
        return cls._gamma(lms @ cls.LMS2RGB)

    @staticmethod
    def from_oklch(oklch):
        L, C, h = oklch[..., 0], oklch[..., 1], oklch[..., 2]
        a, b = C * np.cos(h), C * np.sin(h)
        return np.stack((L, a, b), axis=-1)

    # def clip_gamut(oklch, alpha=0.05):
    #     # https://bottosson.github.io/posts/gamutclipping/#adaptive-%2C-hue-independent
    #     L1, C1 = oklch[..., 0], oklch[..., 1]
    #     Ld = L1 - 0.5
    #     abs_Ld = np.abs(Ld)
    #     e1 = 0.5 + abs_Ld + alpha*C1
    #     L0, C0 = 0.5*(1 + np.sign(Ld)*(e1-np.sqrt(e1**2-2*abs_Ld))), 0.0
    #     t = find_gamut_intersection(...) # !!!
    #     Lt, ct = t*L1 + (1-t)*L0, t*C1 + (1-t)*C0
    #     oklch[..., 0], oklch[..., 1] = Lt, ct
