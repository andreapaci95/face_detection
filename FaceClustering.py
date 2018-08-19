from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import align.detect_face
import facenet
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn.cluster import DBSCAN


class FaceClustering:
    def __init__(self, model_path, input_dir, output_dir):
        self.model = model_path
        self.data_dir = input_dir
        self.out_dir = output_dir
        self.image_size = 160
        self.margin = 32
        self.min_cluster_size = 1
        self.cluster_threshold = 1.0
        self.save_largest_cluster_only = False
        self.gpu_memory_fraction = 1.0

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def process(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.model)

                image_list = load_images_from_folder(self.data_dir, self.image_size)
                images_placeholder = sess.graph.get_tensor_by_name("input:0")
                embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: image_list, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                nrof_images = len(image_list)
                matrix = np.zeros((nrof_images, nrof_images))

                for i in range(nrof_images):
                    for j in range(nrof_images):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                        matrix[i][j] = dist

                # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
                db = DBSCAN(eps=0.60, min_samples=self.min_cluster_size, metric='precomputed')
                db.fit(matrix)
                labels = db.labels_

                cluster_number = len(set(labels)) - (1 if -1 in labels else 0)

                print('Number of clusters:', cluster_number)

                if cluster_number > 0:
                    if self.save_largest_cluster_only:
                        largest_cluster = 0
                        for i in range(cluster_number):
                            if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                                largest_cluster = i
                        print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
                        count = 1
                        for i in np.nonzero(labels == largest_cluster)[0]:
                            misc.imsave(os.path.join(self.out_dir, str(count) + '.png'), image_list[i])
                            count += 1
                    else:
                        print('Saving all clusters')
                        for i in range(cluster_number):
                            count = 1
                            path = os.path.join(self.out_dir, str(i))
                            if not os.path.exists(path):
                                os.makedirs(path)
                            for j in np.nonzero(labels == i)[0]:
                                misc.imsave(os.path.join(path, str(count) + '.png'), image_list[j])
                                count += 1


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder, image_size):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            img = preprocess_image(img, image_size)
            images.append(img)
    return images


def preprocess_image(image, image_size):
    resized = misc.imresize(image, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(resized)
    return prewhitened


def main():
    model_path = "/home/super/datasets/lfw/20170512-110547"
    input_dir = "faces-veloce"
    output_dir = "clusters-veloce"

    face_clustering = FaceClustering(model_path, input_dir, output_dir)
    face_clustering.process()


if __name__ == '__main__':
    main()
