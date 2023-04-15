import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import torch

#import clip
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
#from segment_anything import sam_model_registry, SamPredictor

device='cpu'




#modelClip, preprocessClip = clip.load("ViT-B/32", device=device)


processorCLIPSeg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
modelCLIPSeg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

"""
sam_checkpoint = "sam_vit_b_01ec64.pth"
#sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
"""



def verifDescription(image,description):
    image=cvToPil(image)
    imageClipPrep = preprocessClip(image).unsqueeze(0).to(device)
    text = clip.tokenize([description, "a photo"]).to(device)
    with torch.no_grad():
        image_features = modelClip.encode_image(imageClipPrep)
        text_features = modelClip.encode_text(text)

        logits_per_image, logits_per_text = modelClip(imageClipPrep, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return float(probs[0][0])


def probImageFromDescription(image,description):
    image=cvToPil(image)
    inputs = processorCLIPSeg(text=[description], images=[image], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = modelCLIPSeg(**inputs)

    preds = outputs.logits
    mask=torch.sigmoid(preds)
    return cv2.resize(np.array(mask),image.size)

def pointFromDescription(image,description):
    mask=probImageFromDescription(image,description)
    input_point=np.unravel_index(np.argmax(mask), np.array(mask).shape)
    return np.array([[input_point[1],input_point[0]]])

def basicMaskFromDescription(image,description,seuil):
    mask=probImageFromDescription(image,description)
    gray_image=np.array(mask * 255, dtype = np.uint8)
    (thresh, bw_image) = cv2.threshold(gray_image, seuil, 255, cv2.THRESH_BINARY)
    # fix color format
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
    return bw_image>0

def maskFromPoint(image,input_point):
    predictorSAM.set_image(image)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks[0]

def maskFromDescription(image,description):
    predictorSAM.set_image(image)
    input_point = pointFromDescription(image,description)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks[0]

def show_points(coords, ax, marker_size=375):
    pos_points = coords
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def getMaskedImage(imagecv,mask,color=(0,0,255)):
    plein = np.zeros(imagecv.shape, imagecv.dtype)
    plein[:,:] = color
    maskIm = cv2.bitwise_and(plein, plein, mask=mask.astype(np.uint8))
    return cv2.addWeighted(maskIm, 1, imagecv, 1, 0, imagecv)


def cvToPil(cv2_im):
    cv2_im=cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

def pilToCv(Pil_im):
    image=cv2.cvtColor(np.array(Pil_im), cv2.COLOR_RGB2BGR)
    return image
