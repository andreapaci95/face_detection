import shutil

import cv2
import os


class FrameExtractor:
    def __init__(self, video_file, output_folder):
        self.video = video_file
        self.output_folder = output_folder

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    def extract(self):
        vidcap = cv2.VideoCapture(self.video)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            cv2.imwrite(self.output_folder + os.sep + "%d.jpg" % count, image)
            count += 1
        print ("Extracted", count, "frames from video", self.video)


if __name__ == '__main__':
    fe = FrameExtractor("lento.mp4", "frames-lento")
    fe.extract()
