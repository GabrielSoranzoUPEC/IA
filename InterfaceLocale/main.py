import os
# Pour ne pas afficher les multiples Warnings de tensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#bibliothèque IA
#from seg import *
import nlpQuery as nlp
#from voice import *

from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from cameraFlux import WebcamStream
#import pandas as pd




objet="glasses"
model="None"

# initializing connection with NLP server
nlp.connect()

# initializing and starting multi-threaded webcam input stream 
webcam_stream = WebcamStream(stream_id=0,
                            mask_description=objet,
                            mask_model=model)
webcam_stream.start()


class MyWindow(Tk):
    
    def __init__(self):
        Tk.__init__(self)
               

        self.label_widget = Label(self)
        self.label_widget.pack()

        label1 = Label(self, text="Enter the display mode:")
        label1.pack()
        self.__showMode = StringVar(value='None')
        combobox1 = ttk.Combobox(self, textvariable=self.__showMode)
        combobox1['values'] = ('CLIPSeg', 'CLIPSeg mask only', 'CLIPSeg point only', 'None')
        combobox1.pack()

        button1 = Button(self, text="Change display mode", command=self.changeMode)
        button1.pack()

        self.__target = StringVar(value='Glasses')
        label2 = Label(self, text="Enter the target:")
        label2.pack()
        name2 = Entry(self, textvariable=self.__target)
        name2.focus_set()
        name2.pack()

        label2 = Label(self, text="Enter the action:")
        label2.pack()
        self.__action = StringVar(value='None')
        combobox2 = ttk.Combobox(self, textvariable=self.__action)
        combobox2['values'] = ('Take', 'Drop', 'None')
        combobox2.pack()

        button2 = Button(self, text="Change target and action", command=self.changeTargetAction)
        button2.pack()

        self.__order = StringVar(value='Take the glasses')
        label3 = Label(self, text="Enter an order:")
        label3.pack()
        name3 = Entry(self, textvariable=self.__order)
        name3.focus_set()
        name3.pack()
        button3 = Button(self, text="Change the order", command=self.giveOrder)
        button3.pack()


        self.open_camera()



        #self.geometry( "300x200" )
        self.title( "AI Demo" )

    def changeMode(self):
        webcam_stream.update_mode(self.__showMode.get())


    def giveOrder(self):
        order=self.__order.get()
        res=nlp.findAction(order)
        action=res[0][0]
        score=res[1][0]
        if score>0.1 and (action=="Take" or action=="Drop"):
            target=nlp.findTarget(order,action)
            self.__target.set(target)
            self.__action.set(action)
            '''if action=='Take':
                combobox2.current(0)
            elif action=='Drop':
                combobox2.current(1)
            else:
                combobox2.current(2)'''
        else:
            print("Ordre non compris")



    def open_camera(self):
        frame = webcam_stream.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.label_widget.photo_image = photo_image
        self.label_widget.configure(image=photo_image)
        self.label_widget.after(10, self.open_camera)

    def changeTargetAction(self):
        webcam_stream.update_target(self.__target.get())


app = MyWindow()







app.mainloop()
webcam_stream.stop()
nlp.close()
