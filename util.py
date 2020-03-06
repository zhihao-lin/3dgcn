import numpy as np
import torch
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normal2unit(vertices: "(vertice_num, 3)"):
    """
    Return: (vertice_num, 3) => normalized into unit sphere
    """
    center = vertices.mean(dim= 0)
    vertices -= center
    distance = vertices.norm(dim= 1)
    vertices /= distance.max()
    return vertices

def rotate(points, theta: float, axis: int):
    """Rotate along upward direction"""
    rotate_matrix = torch.eye(3)
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    axises = [0, 1, 2]
    assert  axis in axises
    axises.remove(axis)

    rotate_matrix[axises[0], axises[0]] = cos
    rotate_matrix[axises[0], axises[1]] = -sin
    rotate_matrix[axises[1], axises[0]] = sin
    rotate_matrix[axises[1], axises[1]] = cos
    points = points @ rotate_matrix
    return points

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

class Transform():
    def __init__(self, 
                normal: bool, 
                shift: float = None, 
                scale: float = None, 
                rotate: float = None, 
                axis: int = 0, 
                random:bool= False):

        self.normal = normal
        self.shift = shift
        self.scale = scale
        self.rotate = rotate
        self.axis = axis
        self.random = random

    def __call__(self, points: "(point_num, 3)"):
        if self.normal:
            points = normal2unit(points)

        if self.shift:
            shift = self.shift
            if self.random:
                shift = (torch.rand(3) - 0.5) * self.shift
            points += shift
        
        if self.scale:
            scale = self.scale
            if self.random:
                scale = torch.rand(1).item() * self.scale
            points *= scale
        
        if self.rotate:
            theta = self.rotate
            if self.random: 
                theta = (torch.rand(1).item() - 0.5) * self.rotate
            points = rotate(points, theta, self.axis)

        return points

def test():
    points = torch.randn(1024, 3)
    transform = Transform(normal= True, scale= 10.0, axis= 1, random= True)
    points = transform(points)
    print(points.size())

def test2():
    from sklearn.metrics import jaccard_score
    a = torch.LongTensor(np.random.choice(10, 100))
    b = torch.LongTensor(np.random.choice(10, 100))
    miou_1 = get_miou(a, b, [i for i in range(10)])
    miou_2 = jaccard_score(a, b, average= "macro")
    print(miou_1)
    print(miou_2)

if __name__ == '__main__':
    test()