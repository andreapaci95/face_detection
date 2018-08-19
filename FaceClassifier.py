from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
from shutil import rmtree, copyfile

import facenet
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


def get_file_extension(paths):
    if type(paths) is list:
        extension = "." + paths[0].split('.')[1]
    else:
        extension = "." + paths.split('.')[1]
    return extension


def my_load_model(model, session):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = facenet.get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(session, os.path.join(model_exp, ckpt_file))


class FaceClassifier:
    def __init__(self, pretrained_model, batch_size, image_size):
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.image_size = image_size

    def load_feat_model(self):
        model_feat = find_modelname(self.pretrained_model, ".pkl")
        classifier_filename_exp = os.path.expanduser(model_feat)
        print('Loaded classifier model from file %s' % classifier_filename_exp)
        with open(classifier_filename_exp, 'rb') as infile:
            (model_feat, class_names) = pickle.load(infile)
        return model_feat, class_names

    def load_tf_model(self):
        tf.Graph().as_default()
        sess = tf.Session()

        my_load_model(self.pretrained_model, sess)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        return sess, images_placeholder, embeddings, phase_train_placeholder

    def train(self, dataset_dir, model_output):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(dataset_dir)
                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                facenet.load_model(self.pretrained_model)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(model_output)

                print('Training classifier')
                model = SVC(kernel='rbf', probability=True)
                model.fit(emb_array, labels)

                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

    def predict(self, test_dataset_dir, model, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, classification_dest):
        delete_recreate_dir(classification_dest)

        dataset = facenet.get_dataset(test_dataset_dir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Testing classifier')
        extension = get_file_extension(paths)

        correct = 0
        for index, img_path in enumerate(paths):
            emb_array = self.calculate_embedding_image(embeddings, images_placeholder, img_path,
                                                       phase_train_placeholder, sess)

            predictions = model.predict_proba(emb_array)
            img_gt = img_path.split("/")[-2]
            best_class_indices = np.argmax(predictions, axis=1)
            predicted_class = class_names[best_class_indices[0]]
            save_image_test_folder(img_path, classification_dest, predicted_class, index, extension)

            if predicted_class == img_gt:
                correct += 1

        print("Correct", correct)
        print("All images", len(paths))
        accuracy = correct / float(len(paths))
        print('Accuracy: %.3f' % accuracy)

    def predict_without_gt(self, test_dir, model, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, classification_dest):
        create_dir(classification_dest)
        paths = get_list_of_paths(test_dir)

        print('Testing classifier without gt')
        extension = get_file_extension(paths)

        for index, img_path in enumerate(paths):
            emb_array = self.calculate_embedding_image(embeddings, images_placeholder, img_path,
                                                       phase_train_placeholder, sess)
            filename = img_path.split('/')[1].split('.')[0]
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            predicted_class = class_names[best_class_indices[0]]
            save_image_test_folder(img_path, classification_dest, predicted_class, filename, extension)

    def calculate_embedding_image(self, embeddings, images_placeholder, img_path, phase_train_placeholder, sess):
        image = facenet.load_data([img_path], False, False, self.image_size)
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        return emb_array


def get_list_of_paths(test_dir):
    images_list = []
    for img_path in os.listdir(test_dir):
        images_list.append(os.path.join(test_dir, img_path))
    return images_list


def save_image_test_folder(img_path, classification_dir, predicted_class, index, extension):
    dest_dir = os.path.join(classification_dir, predicted_class)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    copyfile(img_path, os.path.join(dest_dir, str(index) + extension))


def delete_recreate_dir(classification_dest):
    if os.path.exists(classification_dest):
        rmtree(classification_dest)
    os.makedirs(classification_dest)


def create_dir(classification_dest):
    if not os.path.exists(classification_dest):
        os.makedirs(classification_dest)


def find_modelname(model_dir, extension):
    for file in os.listdir(model_dir):
        if file.endswith(extension):
            return os.path.join(model_dir, file)


def main():
    train_dataset_dir = "/home/andreapaci95/datasets/7pixel/train-good"
    test_dataset_dir = "/home/andreapaci95/datasets/7pixel/test-good"
    pretrained_model = "/home/andreapaci95/datasets/7pixel/pretrained-model"
    classifier_filename = "/home/andreapaci95/datasets/7pixel/pretrained-model/modello-addestrato.pkl"
    out_dir = "classification"

    fc = FaceClassifier(pretrained_model, 10, 160)
    fc.train(train_dataset_dir, classifier_filename)

    model_feat, class_names = fc.load_feat_model()
    sess, images_placeholder, embeddings, phase_train_placeholder = fc.load_tf_model()
    fc.predict(test_dataset_dir, model_feat, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, out_dir)


if __name__ == '__main__':
    main()
