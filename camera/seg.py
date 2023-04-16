import numpy as np
import matplotlib.pyplot as plt

# Pour récupérer l'image
import requests

# Pour la manipulation des images
from PIL import Image
import cv2

import torch

# Modèle CLIP d'OpenAI
import clip
# Modèle CLIPSeg pour la Prompt Segmentation
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
# Modèle SAM de Meta
from segment_anything import sam_model_registry, SamPredictor


# Définition du device: idéalement GPU
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Chargement du modèle CLIP
modelClip, preprocessClip = clip.load("ViT-B/32", device=device)





# Chargement du modèle CLIPSeg
processorCLIPSeg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
modelCLIPSeg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")







# Chargement du modèle SAM: doit être localement dans le dossier d'exécution du notebook

# Modèle "léger"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Modèle "lourd"
#sam_checkpoint = "sam_vit_h_4b8939.pth"
#model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)








# On doit ici jongler entre deux formats d'images:
# - Le format PIL est utilisé dans les modèles CLIP et CLIPSeg et dans Matplotlib
# - Le format CV2 est utilisé dans le modèle SAM et pour la gestion des flux vidéos

# Conversion Image au format CV2 -> Image au format PIL
def cvToPil(cv2_im):
    cv2_im=cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

# Conversion Image au format PIL -> Image au format CV2
def pilToCv(Pil_im):
    image=cv2.cvtColor(np.array(Pil_im), cv2.COLOR_RGB2BGR)
    return image







# Dans la fonction suivante on demande au modèle CLIP d'arbitrer probabilistement entre la description donnée dans la variable `description` et une description "passe-partout", ici `"a photo"`. Cela semble assez bien marcher mais peut-être d'autres description "passe-partout" conviendrait mieux, suivant les contextes d'utilisation.

# Fonction (image,texte) -> probabilité de conjonction de l'image et du texte
# Variable Image au format CV2
# Variable description est un string
def verifDescription(image,description):
    # Conversion de l'image au format PIL
    image=cvToPil(image)
    # Preprocessing de l'image et fabrication d'un batch [image]
    imageClipPrep = preprocessClip(image).unsqueeze(0).to(device)
    # Tokenisation du text
    tokens = clip.tokenize([description, "a photo"]).to(device)
    with torch.no_grad():
        #image_features = modelClip.encode_image(imageClipPrep)
        #text_features = modelClip.encode_text(tokens)
        logits_per_image, logits_per_text = modelClip(imageClipPrep, tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return float(probs[0][0])








# Définissons maintenant les fonctions de segmentation utilisant le modèle CLIPSeg. 


# Ce modèle prend en entrée une description et retourne basiquement une segmentation, plus précisément une probabilité de présence pour chaque pixel de l'image de départ.

# Fonction (image,description) -> Image de détection de la description
def probImageFromDescription(image,description):
    image=cvToPil(image)
    inputs = processorCLIPSeg(text=[description], images=[image], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = modelCLIPSeg(**inputs)

    preds = outputs.logits
    mask=torch.sigmoid(preds)
    return cv2.resize(np.array(mask),image.size)






# Fonction (image,description) -> point de l'image de probabilité maximale pour la description
def pointFromDescription(image,description):
    # On récupère l'image des probabilité
    mask=probImageFromDescription(image,description)
    # On prend le point de probabilité maximale
    input_point=np.unravel_index(np.argmax(mask), np.array(mask).shape)
    # On retourne ce point (inversion des x et des y par la fonction unravel_index)
    return np.array([[input_point[1],input_point[0]]])





# Fonction "de confort" pour afficher un point sur une figure matplotlib
def show_points(coords, ax, marker_size=375):
    pos_points = coords
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)





# Fonction (image,description,seuil) -> masque de présence de l'objet conforme à la description
def basicMaskFromDescription(image,description,seuil):
    mask=probImageFromDescription(image,description)
    gray_image=np.array(mask * 255, dtype = np.uint8)
    (thresh, bw_image) = cv2.threshold(gray_image, seuil, 255, cv2.THRESH_BINARY)
    return bw_image>0






# On définit maintenant des fonctions utilisant le modèle SAM de Meta. 
# Ce modèle prend en entrée un point et retourne la segmentation de l'objet présent en ce point.

# Fonction (image,point) -> image mask de l'objet au point fourni
def maskFromPoint(image,input_point):
    # Pré-traitement de l'image par SAM
    predictor.set_image(image)
    # 2 labels possibles dans SAM: 1->le masque doit contenir ce point // 0->le masque ne doit pas contenir ce point
    input_label = np.array([1])
    # Calcul du masque
    mask, score, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    # SAM retourne plusieurs masques possibles: on donne celui de plus grande proba
    return mask.squeeze()





# Pour obtenir un masque à partir d'une description avec SAM on utilise alors le modèle CLIPSeg pour obtenir le point de plus grande probabilité de présence pour ensuite obtenir un masque avec SAM:

# Fonction (image,description) -> image mask correspondant à la description
def maskFromDescription(image,description):
    input_point = pointFromDescription(image,description)
    return maskFromPoint(image,input_point)





# Fonction pour afficher les masques en surrimpression:

# Fonction (image,mask,color) -> image avec masque en surrimpression et de couleur donnée
def getMaskedImage(imagecv,mask,color=(0,0,255)):
    # On crée une image de même dimension que imagecv et de couleur color
    plein = np.zeros(imagecv.shape, imagecv.dtype)
    plein[:,:] = color
    # On colorise notre image msk
    maskIm = cv2.bitwise_and(plein, plein, mask=mask.astype(np.uint8))
    # Onretourne l'addition de maskIm et de imagecv
    return cv2.addWeighted(maskIm, 1, imagecv, 1,0)



