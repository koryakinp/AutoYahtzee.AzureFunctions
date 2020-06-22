import cv2
import math
import os
import glob
import numpy as np
import tensorflow as tf
import tempfile

from sklearn.cluster import MiniBatchKMeans
from tensorflow import keras



def predict(dice_images, model):
    arr = np.array(dice_images)
    arr = np.expand_dims(arr, axis=3)
    softmax = model.predict(arr)
    res = np.argmax(softmax, axis=1) + 1
    prediction = np.argmax(softmax, axis=1) + 1
    res.sort()
    label = ''.join(str(e) for e in res)
    confidence = np.amax(softmax, axis=1) * 100

    return prediction.tolist(), confidence.tolist(), label


def prepare_image_data(filename):
    image = cv2.imread(filename)
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(-1, 1)
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w))

    quant = np.int32(quant)
    quant[quant == quant.min()] = -1
    quant[quant == quant.max()] = 1

    return quant


def compute_conv(image_data, kernels):
    image_data = np.expand_dims(image_data, axis=0)
    image_data = np.expand_dims(image_data, axis=3)
    image_data = np.float32(image_data)

    empty_kernels = np.float32(kernels)

    kernels = np.expand_dims(kernels, axis=0)
    kernels = kernels.transpose(2, 3, 0, 1)

    res = tf.nn.conv2d(image_data, kernels, [1, 1, 1, 1], padding='SAME')
    res = tf.squeeze(res)
    sess = tf.Session()
    with sess.as_default():
        res = res.eval()

    return list(np.int32(res.transpose(2, 0, 1)))


def pad_kernels(kernels):
    max_kernel_w = 0

    for kernel in kernels:
        if kernel.shape[0] > max_kernel_w:
            max_kernel_w = kernel.shape[0]

    temp_kernels = np.zeros((max_kernel_w, max_kernel_w))

    for kernel_id in range(len(kernels)):
        kernel = kernels[kernel_id]
        if kernel.shape[0] < max_kernel_w:
            diff = max_kernel_w - kernel.shape[0]

            if diff % 2 == 0:
                diff = diff/2
            else:
                diff = (diff/2) + 1

            diff = np.int(diff)

            kernels[kernel_id] = np.pad(
                kernel, diff, mode='constant', constant_values=0)

            if kernels[kernel_id].shape[0] > max_kernel_w:
                kernels[kernel_id] = np.delete(kernels[kernel_id], 1, 0)
                kernels[kernel_id] = np.delete(kernels[kernel_id], 1, 1)

    return kernels


def build_kernel(size, angle):
    img = np.full([size, size], 2, dtype=np.uint8)

    img = rotate_image(img, angle)
    img = np.int8(img)

    img[img == 1] = 10
    img[img == 2] = 20
    img[img == 20] = 1
    img[img == 10] = -1

    return img


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    return cv2.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderValue=(0, 0, 0))


def build_kernels(size=33):
    kernels = []

    for angle in range(0, 90, 10):
        kernels.append(build_kernel(size, angle))

    return pad_kernels(kernels)


def get_dice_images_for_human(filename, peaks):
    size = 25
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filenames = []

    for peak in peaks:
        cx = peak[1][0]
        cy = peak[0][0]
        data = image[cy - size:cy + size, cx - size:cx + size]
        jpg = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        filenames.append(jpg)
        cv2.imwrite(jpg.name, data)

    return filenames


def get_dice_images(image, kernels):
    peaks = []
    dices = []
    size = 17
    counter = 0
    original = image.copy()

    while True:
        max_val = -999999

        conv_results = compute_conv(image, kernels)

        for conv_result in conv_results:
            maxx = conv_result.max()

            if maxx > max_val:
                max_val = maxx
                peak = np.where(conv_result == maxx)

        if(max_val < 0 or counter > 10):
            break

        cx = peak[1][0]
        cy = peak[0][0]

        dices.append(original[cy - size:cy + size, cx - size:cx + size].copy())
        peaks.append(peak)
        image = cv2.circle(image, (cx, cy), size, (-1, -1, -1), -1)

        if(max_val < 0 or counter > 10):
            break
        counter += 1
    return dices, peaks


def process_image(filename):
    image = prepare_image_data(filename)
    kernels = build_kernels()
    return get_dice_images(image, kernels)
