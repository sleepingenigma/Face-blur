import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceBlurer():
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Create ROI coordinates
                topLeft = (int(box[0]), int(box[1]))
                bottomRight = (int(box[2]), int(box[3]))
                x, y = topLeft[0], topLeft[1]
                w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

                # Grab ROI with Numpy slicing and blur
                ROI = frame[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(ROI, (51,51), 0) 

                # Insert ROI back into image
                frame[y:y+h, x:x+w] = blur
        except:
            pass
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                self._draw(frame, boxes, probs, landmarks)

            except:
                pass

            # Show the frame
            cv2.imshow('BlurFace', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the app
mtcnn = MTCNN()
fcd = FaceBlurer(mtcnn)
fcd.run()