
from flask import Flask, render_template, Response, request
from cameraFlux import WebcamStream
import pandas as pd
from nlp import *


# initializing and starting multi-threaded webcam input stream 
print("\nInitilisation du flux vidéo")
webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
webcam_stream.start()

ordre=""
action="Neither"

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    global ordre,action

    if request.method == "POST":
        try:
            ordre=str(request.form["ordre"])
        except:
            ordre="Valeur non valide"
            error="Valeur non valide dans le champ ordre.<br>"

        if ordre!="":
            result=findAction(ordre)
            action=result[0][0]
            score=result[1][0]
            print("Action: ",result)
            if score>0.3 and (action=="Take" or action=="Drop"):
                description=findTarget(ordre,action)
                print("Target: ", description)
            else:
                action=""
                description="an object"
        else:
            description = None
            try:
                description = str(request.form["description"])
            except:
                errors += "Valeur non valide dans le champ description.<br>"
            action=None
            try:
                action=str(request.form["action"])
            except:
                errors+= "Valeur non valide dans le champ action.<br>"

        model = None
        try:
            model = str(request.form["model"])
        except:
            errors += "Valeur non valide pour le modèle."
        if description is not None and model is not None:
            if model=="None":
                webcam_stream.update_mask(description,None)
            else:
                webcam_stream.update_mask(description,model)

    if webcam_stream.mask_model=="CLIPSeg":
        checked=["checked","","","",""]
    elif webcam_stream.mask_model=="CLIPSeg mask only":
        checked=["","checked","","",""]
    elif webcam_stream.mask_model=="CLIPSeg point only":
        checked=["","","checked","",""]
    elif webcam_stream.mask_model=="SAM":
        checked=["","","","checked",""]
    else:
        checked=["","","","","checked"]

    if action=="Take":
        checked_act=["checked","",""]
    elif action=="Drop":
        checked_act=["","checked",""]
    else:
        checked_act=["","","checked"]

    return render_template('index.html', description=webcam_stream.mask_description,checked=checked,checked_act=checked_act,ordre=ordre)

def gen():
    while True:
        frame = webcam_stream.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



print("\nLancement de l'application web")
app.run(host='0.0.0.0', debug=False)
