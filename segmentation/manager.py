import os
import sys
sys.path.append('../')
from util import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataset_shapenet import get_valid_labels
from visualize import visualize

def get_miou(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    part_ious = []
    for part_id in valid_labels:
        pred_part = (pred == part_id)
        target_part = (target == part_id)
        I = np.sum(np.logical_and(pred_part, target_part))
        U = np.sum(np.logical_or( pred_part, target_part))
        if U == 0:
            part_ious.append(1)
        else:
            part_ious.append(I/U)
    miou = np.mean(part_ious)
    return miou

class Manager():
    def __init__(self, model, args):
        self.args_info = args.__str__()
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.epoch = args.epoch
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= 10, gamma= 0.5)
        self.loss_function = nn.CrossEntropyLoss()

        self.save = args.save
        self.record_interval = args.interval
        self.record_file = None
        if args.record:
            self.record_file = open(args.record, 'w')
        
        self.out_dir = args.output
        self.best = {"c_miou": 0, "i_miou": 0}

    def update_best(self, c_miou, i_miou):
        self.best["c_miou"] = max(self.best["c_miou"], c_miou)
        self.best["i_miou"] = max(self.best["i_miou"], i_miou)

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')

    def calculate_save_mious(self, iou_table, category_names, labels, predictions):
        for i in range(len(category_names)):
            category = category_names[i]
            pred = predictions[i]
            label =  labels[i]
            valid_labels = get_valid_labels(category)
            miou = get_miou(pred, label, valid_labels)
            iou_table.add_obj_miou(category, miou)

    def save_visualizations(self, dir, category_names, object_ids, points, labels, predictions):
        for i in range(len(category_names)):
            cat = category_names[i]
            valid_labels = get_valid_labels(cat)
            shift = min(valid_labels) * (-1)
            obj_id = object_ids[i]
            point = points[i].to("cpu") 
            label = labels[i].to("cpu") + shift
            pred  = predictions[i].to("cpu") + shift

            cat_dir = os.path.join(dir, cat)
            if not os.path.isdir(cat_dir):
                os.mkdir(cat_dir)
            gt_fig_name = os.path.join(cat_dir, "{}_gt.png".format(obj_id))        
            pred_fig_name = os.path.join(cat_dir, "{}_pred.png".format(obj_id)) 
            visualize(point, label, gt_fig_name)
            visualize(point, pred, pred_fig_name)

    def train(self, train_data, test_data):
        self.record("*****************************************")
        self.record("Hyper-parameters: {}".format(self.args_info))
        self.record("Model parameter number: {}".format(parameter_number(self.model)))
        self.record("Model structure: \n{}".format(self.model.__str__()))
        self.record("*****************************************")

        for epoch in range(self.epoch):
            self.model.train()
            train_loss = 0
            train_iou_table = IouTable()
            for i, (cat_name, obj_ids, points, labels, mask, onehot) in enumerate(train_data):
                points = points.to(self.device)
                labels = labels.to(self.device)
                onehot = onehot.to(self.device)
                out = self.model(points, onehot)

                self.optimizer.zero_grad()
                loss = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1,))     
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                out[mask == 0] = out.min()
                pred = torch.max(out, 2)[1]
                self.calculate_save_mious(train_iou_table, cat_name, labels, pred)

                if self.record_interval and ((i + 1) % self.record_interval == 0):
                    c_miou = train_iou_table.get_mean_category_miou()
                    i_miou = train_iou_table.get_mean_instance_miou()
                    self.record(' epoch {:3} step {:5} | avg loss: {:.3f} | miou(c): {:.3f} | miou(i): {:.3f}'.format(epoch+1, i+1, train_loss/(i + 1), c_miou, i_miou))

            train_loss /= (i+1) 
            train_table_str = train_iou_table.get_string()
            test_loss, test_table_str = self.test(test_data, self.out_dir)
            self.lr_scheduler.step()
            if self.save:
                torch.save(self.model.state_dict(), self.save)

            self.record("==== Epoch {:3} ====".format(epoch + 1))
            self.record("Training mIoU:")
            self.record(train_table_str)
            self.record("Testing mIoU:")
            self.record(test_table_str)
            self.record("* Best mIoU(c): {:.3f}, Best mIoU (i): {:.3f} \n".format(self.best["c_miou"], self.best["i_miou"]))

    def test(self, test_data, out_dir= None):
        if out_dir: 
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

        self.model.eval()
        test_loss = 0
        test_iou_table = IouTable()

        for i, (cat_name, obj_ids, points, labels, mask, onehot) in enumerate(test_data):
            points = points.to(self.device)
            labels = labels.to(self.device)
            onehot = onehot.to(self.device)
            out = self.model(points, onehot)
            loss = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1,))     
            test_loss += loss.item()

            out[mask == 0] = out.min()
            pred = torch.max(out, 2)[1]
            self.calculate_save_mious(test_iou_table, cat_name, labels, pred)
            if out_dir:
                self.save_visualizations(out_dir, cat_name, obj_ids, points, labels, pred)

        test_loss /= (i+1) 
        c_miou = test_iou_table.get_mean_category_miou()
        i_miou = test_iou_table.get_mean_instance_miou()
        self.update_best(c_miou, i_miou)
        test_table_str = test_iou_table.get_string()

        if out_dir:
            miou_file = open(os.path.join(out_dir, "miou.txt"), "w")
            miou_file.write(test_table_str)

        return test_loss, test_table_str
        
class IouTable():
    def __init__(self):
        self.obj_miou = {}
        
    def add_obj_miou(self, category: str, miou: float):
        if category not in self.obj_miou:
            self.obj_miou[category] = [miou]
        else:
            self.obj_miou[category].append(miou)

    def get_category_miou(self):
        """
        Return: moiu table of each category
        """
        category_miou = {}
        for c, mious in self.obj_miou.items():
            category_miou[c] = np.mean(mious)
        return category_miou

    def get_mean_category_miou(self):
        category_miou = []
        for c, mious in self.obj_miou.items():
            c_miou = np.mean(mious)
            category_miou.append(c_miou)
        return np.mean(category_miou)
    
    def get_mean_instance_miou(self):
        object_miou = []
        for c, mious in self.obj_miou.items():
            object_miou += mious
        return np.mean(object_miou)

    def get_string(self):
        mean_c_miou = self.get_mean_category_miou()
        mean_i_miou = self.get_mean_instance_miou()
        first_row  = "| {:5} | {:5} ||".format("Avg_c", "Avg_i")
        second_row = "| {:.3f} | {:.3f} ||".format(mean_c_miou, mean_i_miou)
        
        categories = list(self.obj_miou.keys())
        categories.sort()
        cate_miou = self.get_category_miou()

        for c in categories:
            miou = cate_miou[c]
            first_row  += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(miou)
        
        string = first_row + "\n" + second_row
        return string 

def test():
    from dataset_shapenet import PART_NUM
    import random
    table = IouTable()
    cates = list(PART_NUM.keys())
    for e in range(10):
        for c in cates:
            table.add_obj_miou(c, random.random())
    print(table.get_string()) 

if __name__ == "__main__":
    test()