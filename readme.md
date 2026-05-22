Person Re-id problem:

1- Detection:
	Given a frame, detect all people.

2- Match that person to a gallery of images in feature space.
	-> The gallery itself is extracted from running detection and getting all crops.
	
PIPELINE 01.

1- Seperare pipeline:

	-> Train detector, eg: YOLO, Faster R-CNN
	-> Run detector on images to get person crops. Note the mAP or some other metric to see how good detections are.
	-> Train Re-ID model on crops -> metric learning.
	-> At inference time, you just detect, embed and do nearest neighbour search.


2- Joint Training:
	
	-> The architecure is a shared backbone, Region Proposal Network, ROI, ROI align, detection head + ReID head.
	-> detection loss (bbox coord + classification) + ReID loss (triplet etc.)
	-> Joint backprop

