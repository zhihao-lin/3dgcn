import os
import sys
import argparse
import numpy as np
import pandas as pd
from pyntcloud.io import write_ply

def get_skip_vertice_face_num(path):
    with open(path) as file:
        for i, line in enumerate(file):
            if i == 0 and len(line) > 4: # not just "OFF\n"
                digits = line[3:].split(" ")
                vertices_num = int(digits[0])
                faces_num = int(digits[1])
                return 1, vertices_num, faces_num
            if i == 1:
                digits = line.split(' ')
                vertices_num = int(digits[0])
                faces_num = int(digits[1])
                return 2, vertices_num, faces_num

def get_vertices_faces_from_off_file(path):
    skip, vertices_num, faces_num = get_skip_vertice_face_num(path)
    vertices = np.genfromtxt(path, delimiter= ' ', skip_header= skip, skip_footer= faces_num)
    faces_data = np.genfromtxt(path, dtype= int, delimiter= ' ', skip_header= skip + vertices_num)
    faces = faces_data[:, 1:]
    return vertices, faces

def get_faces_area(v0, v1, v2):
    # v0, v1, v2 are list of vectors [x, y, z] => shape: [length, 3]
    return (0.5) * np.linalg.norm(np.cross((v1 - v0), (v2 - v0)), axis= 1)

def mesh2pointcloud(vertices, faces, points_num= 2048, normalize= False):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    faces_area = get_faces_area(v0, v1, v2)
    faces_prob = faces_area / faces_area.sum()
    face_num = faces.shape[0]
    faces_sample_id = np.random.choice(face_num, size= points_num, p= faces_prob) # (points_num, )
    faces_sample = faces[faces_sample_id] # (points_num, 3)
    
    #  set barycentric coordinates u, v, w => shape: (points_num, )
    u = np.random.rand(points_num, 1)
    v = np.random.rand(points_num, 1)
    exceed_one = (u + v) > 1
    u[exceed_one] = 1 - u[exceed_one]
    v[exceed_one] = 1 - v[exceed_one]
    w = 1 - (u + v)
    
    # sampling
    v0_sample = vertices[faces_sample[:, 0]]
    v1_sample = vertices[faces_sample[:, 1]]
    v2_sample = vertices[faces_sample[:, 2]]
    pointcloud = (v0_sample * u) + (v1_sample * v) + (v2_sample * w)
    
    if normalize:
        center = pointcloud.mean(axis= 0)
        pointcloud -= center
        distance = np.linalg.norm(pointcloud, axis= 1)
        pointcloud /= distance.max()

    pointcloud = pointcloud.astype(np.float32)
    return pointcloud

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', help= "path to ModelNet dataset(e.g. ModelNet40/)", default= None)
    parser.add_argument('-target', help= "path to folder of output points(e.g. ModelNet40_1024_points/)", default= None)
    parser.add_argument('-point_num', type= int, default= 1024, help= "How many points are sampled from each mesh object")
    parser.add_argument('-normal', dest= 'normal', action= 'store_true', help= "Normalize point clouds while sampling")
    parser.set_defaults(normal= False)
    args = parser.parse_args()
    
    source_dir = args.source
    categories_all = [name for name in os.listdir(source_dir) if name not in ['.DS_Store', 'README.txt']]
    target_dir = args.target

    os.mkdir(target_dir)

    for category in categories_all:
        os.mkdir(os.path.join(target_dir, category))
        for mode in ['train', 'test']:
            source_folder = os.path.join(source_dir, category, mode)
            target_folder = os.path.join(target_dir, category, mode)
            os.mkdir(target_folder)

            mesh_names = [os.path.join(source_folder, name) for name in os.listdir(source_folder) if name != '.DS_Store']
            for name in mesh_names:
                vertices, faces = get_vertices_faces_from_off_file(name)
                pointcloud = mesh2pointcloud(vertices, faces, args.point_num, normalize= args.normal)
                # save model 
                model = pd.DataFrame()
                model['x'] = pointcloud[:, 0]
                model['y'] = pointcloud[:, 1]
                model['z'] = pointcloud[:, 2]
                name = name.split('/')[-1]
                target_name = os.path.join(target_folder, name[:-4] + '.ply') 
                write_ply(target_name, points= model)
                
        print('finished category: {}'.format(category))

    print('Finish generating dataset: {}'.format(target_dir))

if __name__ == '__main__':
    main()
