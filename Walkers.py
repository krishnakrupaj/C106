import cv2
import numpy as np

# Create our body classifier
bodClass = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')



# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gry = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = bodClass.detectMultiScale(gry,1.2,3)
    print(bodies)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',cap)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
