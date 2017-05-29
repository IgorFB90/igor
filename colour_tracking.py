# Detects and tracks a green object
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV colour space, then initialize the
# list of tracked points
# green
# lower = (29, 86, 6)
# upper = (64, 255, 255)

# red
# lower = (360, 93, 67)
# upper = (10, 255, 255)

# red battery
lower = (-5, 100, 100)
upper = (15, 255, 255)

pts = deque(maxlen=args["buffer"])

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 60)

fps = camera.get(cv2.CAP_PROP_FPS)

print ('Image Size: '), camera.get(cv2.CAP_PROP_FRAME_WIDTH), ('x'), camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print ('Framerate: '), camera.get(cv2.CAP_PROP_FPS)

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    cv2.putText(frame, ('FPS: %d' % fps), (2, 470), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    # # resize the frame, blur it, Equalize histogram
    # # and convert it to the HSV colour space
    # frame = imutils.resize(frame, width=800)p
    # #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # ### Histogram Equalization
    # hist,bins = np.histogram(frame.flatten(),256,[0,256])
    # cdf = hist.cumsum() #Cumulative Distribution Function
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # frame = cdf[frame]
    # ###
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the colour "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Colour Based Tracking", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
