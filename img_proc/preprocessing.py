# -*- utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def img_endecoding():
    image_raw_data = tf.gfile.FastGFile("backGround.jpg", 'rb').read()

    img_data = tf.image.decode_jpeg(image_raw_data)
    print(type(img_data.eval()))
    print(img_data.eval().ndim)
    print(img_data.eval().dtype)
    print(img_data.eval().size)
    print(img_data.eval().shape)
    # plt.imshow(img_data.eval())
    # plt.show()

    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("backGround2.jpg", 'wb') as f:
        f.write(encoded_image.eval())
    return img_data


def img_proc(img_data):
    # sugggest to convert img to real number domain 0.0-1.0, so as not to lose too much accuracy
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # change size
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    central_cropped = tf.image.central_crop(img_data, 0.5)

    # flip
    flipped_ud = tf.image.flip_up_down(img_data)
    flipped_lr = tf.image.flip_left_right(img_data)
    transpose = tf.image.transpose_image(img_data)
    flipped_rndup = tf.image.random_flip_up_down(img_data)
    flipped_rndlr = tf.image.random_flip_left_right(img_data)

    # brightness
    bright_adjusted = tf.image.adjust_brightness(img_data, -0.5)
    bright_adjusted = tf.image.random_brightness(img_data, 0.5)  # -0.5 - 0.5
    bright_adjusted_clip = tf.clip_by_value(bright_adjusted, 0.0, 1.0)

    # contrast
    contrast_adjusted = tf.image.adjust_contrast(img_data, 0.5)
    contrast_adjusted = tf.image.random_contrast(img_data, 0.5, 5)

    # hue 色相
    hue_adjusted = tf.image.adjust_hue(img_data, 0.1)
    hue_adjusted = tf.image.random_hue(img_data, 0.5)  # 0-0.5

    # saturation 饱和度
    saturation_adjusted = tf.image.adjust_saturation(img_data, -5)
    saturation_adjusted = tf.image.random_saturation(img_data, 0, 5)

    # standardization, N~(0,1)
    adjusted = tf.image.per_image_standardization(img_data)

    # labelling, 输入四维矩阵
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, dtype=tf.float32), 0)
    print(batched.eval().ndim)
    # Ymin, Xmin, Ymax, Xmax
    boxes = tf.constant([[[0.1, 0.5, 0.85, 0.8], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)

    # clip by boxes, 0.4 means at least contain 40% area
    begin, size, box = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes,
        min_object_covered=0.4
    )
    image_with_box = tf.image.draw_bounding_boxes(batched, box)
    distorted_img = tf.slice(img_data, begin, size)

    plt.imshow(result[0].eval())
    plt.show()

    img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
    return img_data


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def process_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox
    )
    distorted_img = tf.slice(image, bbox_begin, bbox_size)
    # resize input image for train, all kinds of interpolation
    distorted_img = tf.image.resize_images(
        distorted_img, [height, width], method=np.random.randint(4)
    )
    # filp img
    distorted_img = tf.image.random_flip_left_right(distorted_img)
    distorted_img = tf.image.random_flip_up_down(distorted_img)
    distorted_img = distort_color(distorted_img, np.random.randint(3))
    return distorted_img


def main():
    # with tf.device('/cpu:0'):
    with tf.Session() as sess:
        img_data = img_endecoding()
        boxes = tf.constant([[[0.1, 0.5, 0.85, 0.8]]])
        # img_data = img_proc(img_data)

        for i in range(6):
            plt.figure(i)
            result = process_for_train(img_data, 500,300,boxes)
            plt.imshow(result.eval())

        plt.show()

        # print(img_data)
        # plt.imshow(trans.eval())
        # plt.show()


if __name__ == '__main__':
    main()
