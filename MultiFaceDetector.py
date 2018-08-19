from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from os.path import basename

import align.detect_face
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class MultiFaceDetector:
    def __init__(self, input_folder, output_folder, margin, image_size):
        self.gpu_memory_fraction = 1.0
        self.minsize = 50  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        output_folder = output_folder + os.sep
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.margin = margin
        self.image_size = image_size

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    def process(self):
        try:
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

                bb_count = 0
                for i in os.listdir(self.input_folder):
                    img = misc.imread(os.path.expanduser(self.input_folder + os.sep + i))
                    img_size = np.asarray(img.shape)[0:2]
                    bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, pnet, rnet, onet,
                                                                      self.threshold,
                                                                      self.factor)

                    for (x1, y1, x2, y2, acc) in bounding_boxes:
                        bb_count += 1

                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(x1 - self.margin / 2, 0)
                        bb[1] = np.maximum(y1 - self.margin / 2, 0)
                        bb[2] = np.minimum(x2 + self.margin / 2, img_size[1])
                        bb[3] = np.minimum(y2 + self.margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')

                        img_filename = basename(i).replace(".png", "") + '_' + str(bb_count) + '.png'
                        output_filename = os.path.join(self.output_folder, img_filename)
                        misc.imsave(output_filename, scaled)

                        w = x2 - x1
                        h = y2 - y1
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        except IOError:
            print('Skipping unreadable image!')

    def process_single(self, img_path):
        try:
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

                bb_count = 0
                img = misc.imread(os.path.expanduser(img_path))
                img_size = np.asarray(img.shape)[0:2]
                bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, pnet, rnet, onet,
                                                                  self.threshold,
                                                                  self.factor)

                for (x1, y1, x2, y2, acc) in bounding_boxes:
                    bb_count += 1

                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(x1 - self.margin / 2, 0)
                    bb[1] = np.maximum(y1 - self.margin / 2, 0)
                    bb[2] = np.minimum(x2 + self.margin / 2, img_size[1])
                    bb[3] = np.minimum(y2 + self.margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')

                    img_filename = basename(img_path).replace(".png", "") + '_' + str(bb_count) + '.png'
                    output_filename = os.path.join(self.output_folder, img_filename)
                    misc.imsave(output_filename, scaled)

                    w = x2 - x1
                    h = y2 - y1
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        except IOError:
            print('Skipping unreadable image!')

    def get_margin(self):
        return self.margin

    def get_image_size(self):
        return self.image_size


def main():
    input_dir = 'frames-veloce'
    output_dir = 'faces-veloce'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mfd = MultiFaceDetector(input_dir, output_dir, 32, 160)
    mfd.process()


if __name__ == '__main__':
    main()
