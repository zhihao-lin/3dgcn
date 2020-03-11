import sys
sys.path.append("../")
from util import parameter_number
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time 

# Object number in dataset:
# catgory    | train | valid | test 
# ----------------------------------
# Airplane   |  1958 |   391 |   341
# Bag        |    54 |     8 |    14
# Cap        |    39 |     5 |    11
# Car        |   659 |    81 |   158
# Chair      |  2658 |   396 |   704
# Earphone   |    49 |     6 |    14
# Guitar     |   550 |    78 |   159
# Knife      |   277 |    35 |    80
# Lamp       |  1118 |   143 |   286
# Laptop     |   324 |    44 |    83
# Motorbike  |   125 |    26 |    51
# Mug        |   130 |    16 |    38
# Pistol     |   209 |    30 |    44
# Rocket     |    46 |     8 |    12
# Skateboard |   106 |    15 |    31
# Table      |  3835 |   588 |   848

PART_NUM = {
    "Airplane": 4,
    "Bag": 2,
    "Cap": 2,
    "Car": 4,
    "Chair": 4,
    "Earphone": 3,
    "Guitar": 3,
    "Knife": 2,
    "Lamp": 4,
    "Laptop": 2,
    "Motorbike": 6,
    "Mug": 2,
    "Pistol": 3,
    "Rocket": 3,
    "Skateboard": 3,
    "Table": 3,
}

TOTAL_PARTS_NUM = sum(PART_NUM.values())

# For calculating mIoU
def get_valid_labels(category: str):
    assert category in PART_NUM
    base = 0
    for cat, num in PART_NUM.items():
        if category == cat:
            valid_labels = [base + i for i in range(num)]
            return valid_labels
        else: 
            base += num

class ShapeNetPart(Dataset):
    def __init__(self, 
                 root:str, 
                 split:str= 'train', 
                 point_num:int= 2500,
                 transform= None):

        super().__init__()
        self.root = root
        self.point_num = point_num
        self.transform = transform
        
        # Set category 
        self.category_id = {}
        with open(os.path.join(root, "synsetoffset2category.txt")) as cat_file:
            for line in cat_file:
                tokens = line.strip().split()
                self.category_id[tokens[1]] = tokens[0]
        self.category_names = list(self.category_id.values())

        # Read split file
        split_file_path = os.path.join(root, "train_test_split", "shuffled_{}_file_list.json".format(split))
        split_file_list = json.load(open(split_file_path, "r"))
        cat_ids = list(self.category_id.keys())
        self.file_list = []     
        for name in split_file_list:
            _, cat_id, obj_id = name.strip().split("/")
            if cat_id in cat_ids:
                self.file_list.append(os.path.join(cat_id, obj_id))

    def get_mask(self, category):
        mask = torch.zeros(TOTAL_PARTS_NUM)
        mask[get_valid_labels(category)] = 1
        mask = mask.unsqueeze(0).repeat(self.point_num, 1)
        return mask

    def get_catgory_onehot(self, category):
        onehot = torch.zeros(len(self.category_names))
        index = self.category_names.index(category)
        onehot[index] = 1
        return onehot

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        cat_id, obj_id = self.file_list[index].split("/")
        category = self.category_id[cat_id]
        
        points = torch.FloatTensor(np.genfromtxt(os.path.join(self.root, cat_id, "points", "{}.pts".format(obj_id))))
        labels = torch.LongTensor(np.genfromtxt(os.path.join(self.root, cat_id, "points_label", "{}.seg".format(obj_id))))
        labels = labels - 1 + get_valid_labels(category)[0]
        sample_ids = torch.multinomial(torch.ones(points.size(0)), num_samples= self.point_num, replacement= True)
        points = points[sample_ids]
        labels = labels[sample_ids]
        if self.transform:
            points = self.transform(points)

        mask = self.get_mask(category)
        onehot = self.get_catgory_onehot(category)
        
        return category, obj_id, points, labels, mask, onehot

def test_model(model, dataset, cuda= "0", bs= 1, point_num= 1024):
    device = torch.device("cuda:{}".format(cuda))
    shapenet = ShapeNetPart(dataset, split= "train", point_num= point_num)
    dataloader = DataLoader(shapenet, batch_size= bs, shuffle= True)
    loss_func = nn.CrossEntropyLoss()
    model = model.to(device)
    
    print("Model parameter #: {}".format(parameter_number(model)))
    for i, (cat_name, obj_id, points, labels, mask, onehot) in enumerate(dataloader):
        print("points size: {}".format(points.size()))
        print("labels size: {}".format(labels.size()))
        print("-----------------------")
        points = points.to(device)
        labels = labels.to(device)
        onehot = onehot.to(device)
        start = time.time()
        out = model(points, onehot)
        print("Finish inference, time: {}".format(time.time() - start))
        loss = loss_func(out.reshape(-1, out.size(-1)), labels.view(-1, ))
        start = time.time()
        loss.backward()
        print("Finish back-prop, time: {}".format(time.time() - start))
        break

def test():
    root = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    part_data = ShapeNetPart(root, split= 'test', point_num= 2048)
    print(len(part_data))
    dataloader = DataLoader(part_data, batch_size= 4, shuffle= True)
    for i, (cat_name, _, points, labels, mask, cat_one_hot) in enumerate(dataloader):
        print("cat name: {:10} | points size: {} | labels size: {} | mask: {} | cat one hot: {}".format(cat_name[0], points.size(), labels.size(), mask.size(), cat_one_hot.size()))
        if i == 10:
            break

if __name__ == "__main__":
    test()