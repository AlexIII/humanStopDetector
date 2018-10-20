# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from datetime import datetime

passingByTimeSec = 0.5
stoppedTimeSec = 2

class Point2D(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __sub__(self, p):
		return Point2D(self.x-p.x, self.y-p.y)
	def norm(self):
		return math.sqrt(self.x**2 + self.y**2)
	def dist(self, p):
		return (self-p).norm()
	def __str__(self):
		return "(%d,%d)"%(self.x, self.y)
	__repr__ = __str__

class Tracker(object):
	def __init__(self, missCount = 3, epsilon = 30):
		self.missCount = missCount
		self.epsilon = epsilon
		self.reset()
	def reset(self):
		self.counts = [] #(position, balance, count)[]
	def update(self, positions):
		#count up balance
		for p in positions:
			self.updateOne(p)
		#count down balance
		self.counts = [(pos, bal-1, cnt, tag) for (pos, bal, cnt, tag) in self.counts]
		#remove those that are lost
		self.counts = [c for c in self.counts if c[1] > -self.missCount]

	def updateOne(self, position):
		e, ind = self.findClosest(position)
		if e is not None:
			pos, bal, cnt, tag = e
			pos = position
			bal += 1
			cnt += 1
			self.counts[ind] = (pos, bal, cnt, tag)
		else: self.counts.append((position, 1, 0, 0))
	def findClosest(self, position):
		minDist = float('inf')
		e = None
		i = 0;
		ind = -1;
		for c in self.counts:
			pos, bal, cnt, tag = c
			d = pos.dist(position)
			if d < self.epsilon and d < minDist:
				minDist = d
				e = c
				ind = i
			i += 1
		return e, ind

def testTracker():
	t = Tracker()
	t.update([Point2D(10,10), Point2D(100,100)])
	print(*t.counts, sep = "; ")
	t.update([Point2D(16,18), Point2D(80,110)])
	print(*t.counts, sep = "; ")
	t.update([Point2D(65,111)])
	print(*t.counts, sep = "; ")
	t.update([Point2D(64,112), Point2D(50,50)])
	print(*t.counts, sep = "; ")
	t.update([Point2D(64,114)])
	print(*t.counts, sep = "; ")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

tracker = Tracker(15, 100)
passedBy = 0
stopped = 0

passedByCnt = 5
stoppedCnt = 30


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	objs = []

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			center = Point2D((startX+endX)/2, (startY+endY)/2)
			if(CLASSES[idx] == "person"):
				objs.append(center)
			cv2.circle(frame, (int(center.x), int(center.y)), 3, COLORS[idx], 2)

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	tracker.update(objs)
	fps.stop()
	#print(str(len(tracker.counts)))
	passedByCnt = int(max(2, passingByTimeSec*fps.fps()))
	stoppedCnt = int(max(20, stoppedTimeSec*fps.fps()))
	print(str(passedByCnt) + " " + str(stoppedCnt))

	for i in range(0, len(tracker.counts)):
		pos, bal, cnt, tag = tracker.counts[i]
		if tag == 0 and cnt == passedByCnt:
			passedBy +=1
			tag += 1
			tracker.counts[i] = (pos, bal, cnt, tag)
		if tag == 1 and cnt == stoppedCnt:
			stopped += 1
			tag += 1
			tracker.counts[i] = (pos, bal, cnt, tag)

	info1 = "passed by: %d"%passedBy
	info2 = "stopped: %d"%stopped
	clr = COLORS[0] #(0, 120, 100)
	cv2.putText(frame, info1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 3)
	cv2.putText(frame, info2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 3)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
