import numpy as np
import cv2
from threading import Thread # library for multi-threading
import time

from seg import *

# Class pour gérer le multi-threading du flux video
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("Nombre de frame par seconde: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)        # self.stopped is initialized to False 
        self.stopped = True        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()    # method to return latest read frame 
    def read(self):
        return self.frame    # method to stop reading frames 
    def stop(self):
        self.stopped = True

# initializing and starting multi-threaded webcam input stream 
webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
webcam_stream.start()
start = time.time()



description="a pen"


while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read()   


    if frame is not None:
        basicMask=basicMaskFromDescription(frame,description,150)
        frame=getMaskedImage(frame,basicMask)

        
    cv2.imshow('frame' , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    
end = time.time()
webcam_stream.stop() # stop the webcam stream
cv2.destroyAllWindows()
