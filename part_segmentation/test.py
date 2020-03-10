
import sys
sys.path.append("../")
from util import parameter_number
from dataset_shapenet import ShapeNetPart
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

def test_model(model, dataset, cuda= "0", bs= 1, point_num= 1024):
    device = torch.device("cuda:{}".format(cuda))
    shapenet = ShapeNetPart(dataset, split= "train", point_num= point_num)
    dataloader = DataLoader(shapenet, batch_size= bs, shuffle= True)
    loss_func = nn.CrossEntropyLoss()
    model = model.to(device)
    
    print("Model parameter #: {}".format(parameter_number(model)))
    for i, (cat_name, _, points, labels, mask, onehot) in enumerate(dataloader):
        print("points size: {}".format(points.size()))
        print("labels size: {}".format(labels.size()))
        print("-----------------------")
        points = points.to(device)
        labels = labels.to(device)
        onehot = onehot.to(device)
        start = time.time()
        out = model(points, onehot)
        print("Finish inference, time: {}".format(time.time() - start))
        labels = labels.view(-1, )
        out = out.reshape(-1, out.size(-1))
        loss = loss_func(out, labels)
        start = time.time()
        loss.backward()
        print("Finish back-prop, time: {}".format(time.time() - start))
        break