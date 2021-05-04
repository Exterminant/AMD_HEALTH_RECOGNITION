import os
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from t_utils.engine import train_one_epoch
from t_utils import utils
from t_utils import transforms as T


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    label = data[data["filename"] == filename][["class"]].values
    return boxes_array, label


def merge(labels):
    new_labels = []
    for label in labels:
        if label == "AMD":
            label = 1
            new_labels.append(label)
        elif label == "health":
            label = 2
            new_labels.append(label)
    return [new_labels]


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "AMD_new/")))
        self.path_to_data_file = data_file
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "AMD_new/"+self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list, labels = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        num_objs = len(box_list)
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        new_labels = merge(labels)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(new_labels[0])
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            return img, target

    def __len__(self):
        return len(self.imgs)


dataset = AbstractDataset(root="./", data_file="labels.csv", transforms=get_transform(train=False))
indices = torch.randperm(len(dataset)).tolist()
print(dataset.__getitem__(1))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                          collate_fn=utils.collate_fn, pin_memory=True)
print("We have: {} examples, {} are training".format(len(indices), len(dataset)))
img_ids = set([i[:-4] for i in os.listdir("AMD_new")])
xmls_ids = set([i[:-4] for i in os.listdir("xmls")])
diff = img_ids.symmetric_difference(xmls_ids)
print(diff)
device = torch.device('cuda')
model = get_model(3)
print(model)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 10
for epoch in range(num_epochs):
   # train for one epoch, printing every 10 iterations
   train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
   # update the learning rate
   lr_scheduler.step()

torch.save(model.state_dict(), "sleeve_t.pth")
