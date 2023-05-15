import os
# Pour ne pas afficher les multiples Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#bibliothèque IA
from seg import *
from nlp import *
from voice import *

from tkinter import *
import cv2
from PIL import Image, ImageTk
from cameraFlux import WebcamStream
#import pandas as pd




objet="a pen"
model=None


# initializing and starting multi-threaded webcam input stream 
webcam_stream = WebcamStream(stream_id=0,
                            mask_description=objet,
                            mask_model=model)
webcam_stream.start()

app = Tk()
app.bind('<Escape>', lambda e: app.quit())
label_widget = Label(app)
label_widget.pack()

def open_camera():
    frame = webcam_stream.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)

#button1 = Button(app, text="Open Camera", command=open_camera)
#button1.pack()
open_camera()

app.mainloop()
webcam_stream.stop()
