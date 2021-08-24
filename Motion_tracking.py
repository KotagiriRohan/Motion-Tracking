
import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)
_, frame1 = vid.read()
_, frame2 = vid.read()

while vid.isOpened():
    frame = cv.absdiff(frame1, frame2)
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(5,5),0)
    _,thresh = cv.threshold(blur, 10, 255, cv.THRESH_BINARY)
    dialate = cv.dilate(thresh, None , iterations=3)
    Contours, _ = cv.findContours(dialate, cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
    for contour in Contours:
        (x,y,w,h) = cv.boundingRect(contour)
        
        if cv.contourArea(contour) > 3000:
            cv.rectangle(frame1, (x,y) , (x+w,y+h) , (0,255,0) , 2)
            cv.putText(frame1, "Status: {}".format("Movement"),(10,20),cv.FONT_HERSHEY_DUPLEX, 1,(0,255,0),3)
        #cv.drawContours(frame2, Contours, -1, (0,255,0), 2)
    cv.imshow("display", frame1)
    
    frame1 = frame2
    _, frame2 = vid.read()
    if(cv.waitKey(1) & 0xFF == ord('q')):
        break
cv.destroyAllWindows()