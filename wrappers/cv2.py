"""Reviewed OpenCV image operations; no GUI, devices, models or host file IO."""
import io
import math
import operator
import cv2
import numpy as np
from PIL import Image
from .api import select
from .common import is_view, iterate, assign
from .files import source, save
from .numpy import checked_size

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
FUNCTIONS = '''rotate flip split merge mixChannels extractChannel insertChannel
    GaussianBlur blur medianBlur bilateralFilter boxFilter sqrBoxFilter filter2D sepFilter2D
    Sobel Scharr Laplacian Canny threshold adaptiveThreshold erode dilate morphologyEx
    findContours drawContours contourArea arcLength boundingRect minAreaRect boxPoints
    minEnclosingCircle approxPolyDP convexHull isContourConvex moments HuMoments pointPolygonTest
    connectedComponents connectedComponentsWithStats
    line rectangle circle ellipse polylines fillPoly fillConvexPoly putText getTextSize clipLine
    bitwise_and bitwise_or bitwise_xor bitwise_not add subtract multiply divide addWeighted
    absdiff convertScaleAbs normalize inRange minMaxLoc mean meanStdDev sumElems countNonZero
    sqrt exp log pow magnitude phase cartToPolar polarToCart
    getRotationMatrix2D getAffineTransform getPerspectiveTransform invertAffineTransform
    transform perspectiveTransform equalizeHist compareHist
    distanceTransform integral integral2 integral3 matchTemplate
    findNonZero fitLine fitEllipse fitEllipseAMS fitEllipseDirect'''
PREFIXES = ('COLOR_', 'IMREAD_', 'IMWRITE_', 'INTER_', 'BORDER_', 'THRESH_',
            'MORPH_', 'RETR_', 'CHAIN_APPROX_', 'FONT_', 'LINE_', 'ROTATE_',
            'NORM_', 'CMP_', 'DIST_', 'CC_STAT_', 'CV_', 'HISTCMP_', 'TM_')


def size(shape, dtype):
    checked_size(tuple(operator.index(n) for n in shape), dtype)


def checked(name, function):
    def call(*args, **kwargs):
        src = args[0] if args else kwargs.get('src', kwargs.get('image'))
        kernels = {'GaussianBlur': 1, 'blur': 1, 'medianBlur': 1, 'boxFilter': 2, 'sqrBoxFilter': 2}
        if name in kernels:
            index = kernels[name]
            ksize = args[index] if len(args) > index else kwargs['ksize']
            shape = (ksize, ksize) if isinstance(ksize, int) else tuple(ksize)
            if name == 'GaussianBlur' and 0 in shape:
                sx = kwargs.get('sigmaX', args[2] if len(args) > 2 else 0)
                sy = kwargs.get('sigmaY', args[4] if len(args) > 4 else sx) or sx
                if any(not math.isfinite(sigma) or sigma < 0 for sigma in (sx, sy)):
                    raise ValueError('sigma must be finite and nonnegative')
                shape = tuple(n or math.ceil(sigma * 8 + 1) for n, sigma in zip(shape, (sx, sy)))
            size(shape, 'float64')
        if name in ('Sobel', 'Scharr', 'Laplacian', 'filter2D', 'sepFilter2D', 'boxFilter', 'sqrBoxFilter'):
            size(src.shape, 'float64')
        if name in ('connectedComponents', 'connectedComponentsWithStats'):
            size((src.size, 8), 'float64')
        if name == 'matchTemplate':
            image = args[0] if args else kwargs['image']
            template = args[1] if len(args) > 1 else kwargs['templ']
            size((abs(image.shape[0]-template.shape[0])+1, abs(image.shape[1]-template.shape[1])+1), 'float32')
        return function(*args, **kwargs)
    return call


def sized(function, *, warp=False):
    def call(src, *args, **kwargs):
        dsize = args[1 if warp else 0] if len(args) > int(warp) else kwargs['dsize']
        width, height = (0, 0) if dsize is None else map(operator.index, dsize)
        if not warp and (width == 0 or height == 0):
            fx = kwargs.get('fx', args[2] if len(args) > 2 else 0)
            fy = kwargs.get('fy', args[3] if len(args) > 3 else 0)
            width, height = round(src.shape[1] * fx), round(src.shape[0] * fy)
        size((height, width, *src.shape[2:]), src.dtype)
        return function(src, *args, **kwargs)
    return call


def exports(fs):
    def decode(buf, flags=cv2.IMREAD_COLOR):
        if flags >= 0 and flags & cv2.IMREAD_LOAD_GDAL:
            raise ValueError('GDAL image loading is not supported')
        data = bytes(buf) if isinstance(buf, (bytes, bytearray)) else np.asarray(buf, dtype=np.uint8).tobytes()
        try:
            with Image.open(io.BytesIO(data)) as image:
                size((image.height, image.width, 4), 'float64')
        except (Image.UnidentifiedImageError, OSError):
            return None
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)

    def read(filename, flags=cv2.IMREAD_COLOR):
        try:
            return decode(source(fs, filename).getvalue(), flags)
        except FileNotFoundError:
            return None

    def encode(ext, img, params=None):
        if not isinstance(ext, str) or ext.lower() not in EXTENSIONS:
            raise ValueError('Unsupported image format')
        return cv2.imencode(ext.lower(), img, [] if params is None else params)

    def write(filename, img, params=None):
        from pathlib import PurePosixPath
        ok, data = encode(PurePosixPath(str(filename)).suffix, img, params)
        if ok:
            save(fs, filename, lambda stream: stream.write(data.tobytes()))
        return ok

    def color(src, code, dst=None, dstCn=0):
        size((*src.shape[:2], max(4, operator.index(dstCn))), src.dtype)
        return cv2.cvtColor(src, code, dst=dst, dstCn=dstCn)

    def kernel(shape, ksize, anchor=(-1, -1)):
        size(tuple(ksize), 'uint8')
        return cv2.getStructuringElement(shape, ksize, anchor)

    def histogram(images, channels, mask, histSize, ranges, hist=None, accumulate=False):
        size(histSize, 'float32')
        return cv2.calcHist(images, channels, mask, histSize, ranges, hist=hist, accumulate=accumulate)

    def border(src, top, bottom, left, right, borderType, dst=None, value=None):
        size((src.shape[0] + top + bottom, src.shape[1] + left + right, *src.shape[2:]), src.dtype)
        return cv2.copyMakeBorder(src, top, bottom, left, right, borderType, dst=dst, value=value)

    constants = {name: value for name, value in vars(cv2).items()
                 if name.startswith(PREFIXES) and type(value) in (int, float)}
    return {**constants, **{name: checked(name, function) for name, function in select(vars(cv2), FUNCTIONS).items()},
            'imread': read, 'imwrite': write, 'imdecode': decode, 'imencode': encode,
            'cvtColor': color, 'resize': sized(cv2.resize),
            'warpAffine': sized(cv2.warpAffine, warp=True),
            'warpPerspective': sized(cv2.warpPerspective, warp=True),
            'getStructuringElement': kernel, 'copyMakeBorder': border, 'calcHist': histogram}


def members(value, fs, name=None):
    return {}
