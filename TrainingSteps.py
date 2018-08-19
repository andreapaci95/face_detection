import os
import shutil

from FaceClustering import FaceClustering
from FrameExtractor import FrameExtractor
from MultiFaceDetector import MultiFaceDetector


def main():
    folder = "/home/andreapaci95/datasets/7pixel/videos/train"
    for file in os.listdir(folder):
        video_name = file.split(".")[-2]
        video_file = folder + os.sep + video_name + ".mp4"
        frames_folder = "frames-" + video_name
        faces_folder = "faces-" + video_name
        clusters_folder = "clusters-" + video_name
        model_path = "/home/andreapaci95/datasets/7pixel/pretrained-model"

        fe = FrameExtractor(video_file, frames_folder)
        fe.extract()

        mfd = MultiFaceDetector(frames_folder, faces_folder, 32, 160)
        mfd.process()

        fc = FaceClustering(model_path, faces_folder, clusters_folder)
        fc.process()

        shutil.rmtree(frames_folder)
        shutil.rmtree(faces_folder)

        # After all of this step you have to create the dataset manually, then train a classifier


if __name__ == '__main__':
    main()
