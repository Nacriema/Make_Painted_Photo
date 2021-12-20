#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sep 22 22:38:25 2021

@author: Nacriema

Refs:
This is a case study from: P. Litwinowicz’s SIGGRAPH’97 paper “Processing Images and Videos for an Impressionist Effect”

Basically, I will go through each step as follow:

* Step 1: Stroke Scan-Conversion.
"""
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal, ndimage

# Read and display image
image_source = './Image/landscape_1.jpg'
image_rgb = imageio.imread(image_source)

# NOTICE: PIL JUST DRAW ON IT DATA STRUCTURE
result_img = Image.open(image_source)
h, w, c = image_rgb.shape
# Make grid
n_points = 200
x = np.linspace(20, w - 20, n_points).astype(int)
y = np.linspace(20, h - 20, n_points).astype(int)
xv, yv = np.meshgrid(x, y)

z = np.stack((xv, yv), axis=-1).reshape(-1, 2)

# Make a stroke line
stroke_length = np.random.randint(low=3, high=20)
stroke_angle = 45.  # 45 degree

# TODO: NOTICE ABOUT THE INDEXES
#  in build-in of plt ----> x and | for y and image origin is on top left then I should follow this


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def rgb2hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# TODO: NEED MORE CODE REFRESHMENT STEPS
# Get the code from image_gradient
# Merge variable later


I = imageio.imread(image_source)
# TODO: CV2 !!! Eliminate it
I_gray = rgb2gray(I)
I_gray = ndimage.gaussian_filter(I_gray, sigma=0.8, truncate=3)

sobel = np.array([
    [1 + 1j, 0 + 2j, -1 + 1j],
    [2 + 0j, 0 + 0j, -2 + 0j],
    [1 - 1j, 0 - 2j, -1 - 1j]
])

# 1. Compute image gradient at pixel
grad = signal.convolve2d(I_gray, sobel, boundary='symm', mode='same')
grad_x = grad.real
grad_y = grad.imag

# 2. Compute gradient magnitude at each pixel(in the range 0-255)
grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
grad_mag = grad_mag / np.amax(grad_mag) * 255.
grad_mag = grad_mag.astype(np.uint8)

# 3. If magnitude > threshold then label it as an edge pixel
unique, count = np.unique(grad_mag, return_counts=True)
# print(dict(zip(unique, count)))
T = 50  # T=10 is good
mask = grad_mag > T
# TODO: Let the user define the thresh hold !!! I must inform the histogram of it to the user.

# plt.imshow(mask)
# plt.show()

# TODO: COMPUTE THE GRADIENT ORIENTATION
grad_ori = np.arctan2(grad_x, grad_y) * 180. / np.pi  # In degree

# TODO: Make a function that generate a line (2 point pair) given a point, length of line and angle
'''
Base on the plt coord, I use then :

O -------->x
|
|           /  line
|          /
v y       x point (a, b)
        /
'''


# USE BRESSEN HAM LINE DRAW to trace the right orientation from the src to destination point
def get_line_bound(x0, y0, x1, y1, mask):
    """
    x0, y0 is the origin, x1, y1 is the boundary points
    mask is the edge image detected by Sobel operator
    It returns the end point from x0 y0 to x1 y1 direction: if during trace, it reaches the edge then stop at that pixel
    """
    first_check = True
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if first_check:
            result = (x0, y0)
            first_check = False

        if mask[y0, x0]:
            result = (x0, y0)
            break
        if x0 == x1 and y0 == y1:
            result = (x0, y0)
            break
        result = (x0, y0)
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return result


def get_point_pair(origin, length, mask, angle=45.):
    angle = angle / 180. * np.pi
    x0, y0 = origin
    c = np.tan(-angle)
    x1 = x0 + c * length / (2 * np.sqrt(1 + c ** 2))
    y1 = y0 + length / (2 * np.sqrt(1 + c ** 2))
    x2 = x0 - c * length / (2 * np.sqrt(1 + c ** 2))
    y2 = y0 - length / (2 * np.sqrt(1 + c ** 2))

    # Update x1, y1, x2, y2
    x1, y1 = get_line_bound(int(x0), int(y0), int(x1), int(y1), mask)
    x2, y2 = get_line_bound(int(x0), int(y0), int(x2), int(y2), mask)
    rs = np.array([x1, y1, x2, y2])
    return rs


# TEST DRAW ON THE MASK INSTEAD
draw = ImageDraw.Draw(result_img)

# TODO: I TEST ON THE MASK INSTEAD, I MOVED IT TO THE STROKE_CLIPPING.PY
# mask_PIL = Image.fromarray(np.uint8(mask)*255)  # Convert to 0-255 range
# draw = ImageDraw.Draw(mask_PIL)

for i in range(len(z)):
    coord = z[i]
    color_rgb = image_rgb[coord[1], coord[0]]
    # hex_string = colors.to_hex(color_rgb / 255.)
    hex_string = rgb2hex(tuple(color_rgb))
    stroke_length = np.random.randint(low=20, high=40)
    a1, b1, a2, b2 = get_point_pair(z[i], length=stroke_length, mask=mask, angle=90. - grad_ori[coord[1], coord[0]])
    stroke_width = np.random.randint(low=5, high=15)
    draw.line((a1, b1, a2, b2), fill=hex_string, width=stroke_width)

# TODO: SHOW THE STROKE IMAGE
# result_img.show()


h, w, c = image_rgb.shape
mask = mask[:, :, np.newaxis]

result_img_np = np.asarray(result_img)
edge_rgb = mask * image_rgb
stroke_without_edge = ~mask * result_img_np

# Stroke Clipping Results
stroke_clipping_rs = stroke_without_edge + edge_rgb
plt.imshow(mask)
plt.show()
plt.imshow(result_img_np)
plt.show()
