import numpy as np
import torch

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

def rotate(points, degree: float, axis: int):
    """Rotate along upward direction"""
    rotate_matrix = torch.eye(3)
    theta = (degree/360)*2*np.pi
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
                shift = (torch.rand(3)*2 - 1) * self.shift
            points += shift
        
        if self.scale:
            scale = self.scale
            points *= scale
        
        if self.rotate:
            degree = self.rotate
            if self.random: 
                degree = (torch.rand(1).item()*2 - 1) * self.rotate
            points = rotate(points, degree, self.axis)

        return points

def test():
    points = torch.randn(1024, 3)
    transform = Transform(normal= True, scale= 10.0, axis= 1, random= True)
    points = transform(points)
    print(points.size())

if __name__ == '__main__':
    test()