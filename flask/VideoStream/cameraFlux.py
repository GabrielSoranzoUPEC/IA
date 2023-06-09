import cv2
from threading import Thread # library for multi-threading

print("Importation IA functions...")
# Bibliothèque personelle des fonction d'IA
from seg import *



# Class pour gérer le multi-threading du flux video
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0,mask_description='glasses',mask_model="CLIPSeg"):
        self.stream_id = stream_id # default is 0 for main camera 
        # opening video capture stream 
        self.vcap  = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        # hardware fps
        fps_input_stream = int(self.vcap.get(5))
        print("Nombre de frame par seconde: {}".format(fps_input_stream))
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        # daemon threads run in background
        self.t.daemon = True
        # Fix the mask
        self.mask_description=mask_description
        self.mask_model=mask_model
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()

    def update_mask(self,mask_description, mask_model):
        self.mask_description=mask_description
        self.mask_model=mask_model

    # method passed to thread to read next available frame
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        basicMask=None
        if self.mask_model=="CLIPSeg":
            basicMask=basicMaskFromDescription(self.frame,self.mask_description,150)
            image=getMaskedImage(self.frame,basicMask)
        elif self.mask_model=="CLIPSeg mask only":
            probImage=probImageFromDescription(self.frame,self.mask_description)
            image=np.array(probImage * 255, dtype = np.uint8)
        elif self.mask_model=="CLIPSeg point only":
            coord=tuple((pointFromDescription(self.frame,self.mask_description)[0]))
            image = cv2.circle(self.frame, coord, radius=10, color=(0, 0, 255), thickness=-1)
        elif self.mask_model=="SAM":
            basicMask=maskFromDescription(self.frame,self.mask_description)
            image=getMaskedImage(self.frame,basicMask)
        elif self.mask_model==None:
            image=self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    # method to stop reading frames
    def stop(self):
        self.stopped = True



