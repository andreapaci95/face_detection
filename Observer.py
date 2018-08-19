import time
import os

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


class FaceDetectionHandler(PatternMatchingEventHandler):

    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        os.system("python TestSteps.py " + event.src_path + " faces first-pip-test")

    def on_created(self, event):
        self.process(event)


if __name__ == '__main__':
    args = "hotfolder"
    observer = Observer()
    observer.schedule(FaceDetectionHandler(), path=args)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
