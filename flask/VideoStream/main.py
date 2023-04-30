
from flask import Flask, render_template, Response, request
from cameraFlux import WebcamStream
import pandas as pd
from nlp import *
from voice import *


# initializing and starting multi-threaded webcam input stream 
print("\nInitilisation du flux vidéo")
webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
webcam_stream.start()

ordre="Take the glasses."
action="Take"
audio_file="/home/gs/Info/IA/flask/VideoStream/audio.mp3"

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    global ordre,action,audio_file

    if request.method == "POST":
        errors=""
        try:
            audio_file=str(request.form["audio_file"])
        except:
            audio_file=""
            error="Valeur non valide dans le champ audio_file<br>"

        if audio_file!="":
            ordre=audioToText(audio_file)
        else:
            try:
                ordre=str(request.form["ordre"])
            except:
                ordre=""
                error="Valeur non valide dans le champ ordre.<br>"

        if ordre!="":
            result=findAction(ordre)
            action=result[0][0]
            score=result[1][0]
            print("Action: ",result)
            if score>0.1 and (action=="Take" or action=="Drop"):
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

        if errors!="":
            print(errors)

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

    return render_template('index.html', description=webcam_stream.mask_description,checked=checked,
                           checked_act=checked_act,ordre=ordre,audio_file=audio_file)

def gen():
    while True:
        frame = webcam_stream.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    for filename, file in request.files.items():
        print(filename, file)

    if 'audio' not in request.files:
        print("Erreur de la reception de l'enregistrement audio")
        return 'No file uploaded.', 400

    print("Reception de l'enregistrement audio")
    file = request.files['audio']
    file.save('./audio.mp3')

    return 'File uploaded successfully.', 200


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



print("\nLancement de l'application web")
app.run(host='0.0.0.0', debug=False)
