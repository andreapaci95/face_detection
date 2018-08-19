import sys
import warnings

from FaceClassifier import FaceClassifier
from FrameExtractor import FrameExtractor
from MultiFaceDetector import MultiFaceDetector

warnings.filterwarnings("ignore")


def main():
    model_path = "/home/andreapaci95/datasets/7pixel/pretrained-model"

    video_file = "/home/andreapaci95/datasets/7pixel/videos/test/test7.mp4"
    frames_folder = "frames"
    faces_folder = "faces"
    classification_dir = "first-pip-test"

    fe = FrameExtractor(video_file, frames_folder)
    fe.extract()

    mfd = MultiFaceDetector(frames_folder, faces_folder, 32, 160)
    mfd.process()

    fc = FaceClassifier(model_path, 90, 160)
    model_feat, class_names = fc.load_feat_model()
    sess, images_placeholder, embeddings, phase_train_placeholder = fc.load_tf_model()

    fc.predict_without_gt(faces_folder, model_feat, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, classification_dir)


if __name__ == '__main__':
    main()
