import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor as tens
from PIL import Image, ImageDraw
import os
import pandas as pd


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def predict(img, model, device):
    img_tens = tens()(img)
    cuda_tens = img_tens.cuda()
    cuda_tens.to(device)
    image = Image.fromarray(cuda_tens.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    draw = ImageDraw.Draw(image)
    name = np.random.randint(100, size=(10,))
    name = [str(n) for n in name]
    fin_name = ""
    label = ""
    for i in name:
        fin_name = "".join(i)
    with torch.no_grad():
        prediction = model([cuda_tens])
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        if prediction[0]["labels"][element] == 1:
            if score > 0.75:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
                draw.text((boxes[0], boxes[1]), text="AMD"+str(score), fill=(0, 0, 0))
                label = "AMD"
        elif prediction[0]["labels"][element] == 2:
            if score > 0.75:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="green", width=3)
                draw.text((boxes[0], boxes[1]), text="health"+str(score), fill=(0, 0, 0))
                label = "health"
    return label


device = torch.device("cuda")
loaded_model = get_model(num_classes=3)
loaded_model.load_state_dict(torch.load('sleeve_t.pth'))
loaded_model.to(device)
loaded_model.eval()
ss = pd.read_csv("test.csv")
ass = []
count = 0
for i in range(len(ss)):
    row = ss.loc[i]
    ground_truth = row["class_label"]
    image_name = row["image"]
    image = Image.open("test/"+image_name)
    label = predict(image, loaded_model, device)
    if label == ground_truth:
        ass.append(1)
    count +=1


accuracy = len(ass)/count
print("Right-predicted images", len(ass))
print("Total images", count)
print("accuracy", accuracy)








