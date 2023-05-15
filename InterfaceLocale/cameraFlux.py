import cv2
from threading import Thread # library for multi-threading




font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,10)
fontScale              = 0.5
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2




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
        image=self.frame;
        if self.mask_model!=None:
            probImage=probImageFromDescription(image,self.mask_description)
            probMax=np.max(probImage)
            if self.mask_model=="CLIPSeg":
                basicMask=basicMaskFromProb(probImage,150)
                image=getMaskedImage(image,basicMask)
            elif self.mask_model=="CLIPSeg mask only":
                image=np.array(probImage * 255, dtype = np.uint8)
            elif self.mask_model=="CLIPSeg point only":
                if probMax>0.25:
                    coord=np.unravel_index(np.argmax(probImage), np.array(probImage).shape)
                    coord=(coord[1],coord[0])
                    image = cv2.circle(image, coord, 
                                   radius=10, color=(0, 255, 0), thickness=-1)
        
            cv2.putText(image,'Target : '+self.mask_description,(10,20),
                        font,fontScale,fontColor,thickness,lineType)
            if probMax>0.25:
                color=(0,255,0)
            else:
                color=(0,0,255)
            cv2.putText(image,'Detection max prob : '+str(probMax),(10,35),
                        font,fontScale,color,thickness,lineType)
        return image

    # method to stop reading frames
    def stop(self):
        self.stopped = True



