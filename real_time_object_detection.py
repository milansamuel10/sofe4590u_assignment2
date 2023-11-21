import numpy as np
import argparse
import imutils
import time
import cv2
import os

prototxt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MobileNetSSD_deploy.prototxt.txt")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MobileNetSSD_deploy.caffemodel")
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person-bicycle-car-detection.mp4")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default=prototxt_path,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default=model_path,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak predictions")
ap.add_argument("-v", "--video", default=video_path,
                help="path to video file")
args = vars(ap.parse_args())

# Initialize the list of class labels and corresponding colors
CLASSES = ["background", "bicycle", "bus", "person", "motorbike", "car", "train"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model from Caffe
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Open the video stream
vs = cv2.VideoCapture(args["video"])
if not vs.isOpened():
    print("[ERROR] Couldn't open video file.")
    exit()

fps = vs.get(cv2.CAP_PROP_FPS)
print("[INFO] Frames per second: {:.2f}".format(fps))

# start the timer
start_time = time.time()
frame_count = 0

# Loop over the frames from the video stream
while True:
    ret, frame = vs.read()
    if not ret:
        print("[INFO] End of video stream.")
        break
    frame = imutils.resize(frame, width=400)

    # Preprocess the frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, (1 / 127.5), (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(predictions[0, 0, i, 1])

            if 0 <= idx < len(CLASSES):

                box = predictions[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print(f"Frame {frame_count} - Object detected: {label}")

                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    frame_count += 1

end_time = time.time()
fps = vs.get(cv2.CAP_PROP_FPS)
elapsed_time = end_time - start_time

# Release the video stream and clean up
vs.release()
cv2.destroyAllWindows()

print("[INFO] Total Frames: {:.2f}".format(frame_count))
print("[INFO] Total Elapsed Time: {:.2f}".format(elapsed_time))
print("[INFO] Approximate FPS: {:.2f}".format(frame_count / elapsed_time))