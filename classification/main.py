import argparse
import torch
from torch.utils.data import DataLoader
from model_gcn3d import GCN3D
from model_dgcnn import DGCNN
from model_pointnet import PointNetCls
from dataset_modelnet import ModelNet_pointcloud
from manager import Manager
import sys
sys.path.append("..")
from util import Transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', help= 'train/test', default= 'train')
    parser.add_argument('-cuda', help= 'Cuda index', default= '0')
    parser.add_argument('-epoch', type= int, default= 100)
    parser.add_argument('-lr', help= 'Learning Rate', type= float, default= 1e-4)
    parser.add_argument('-bs', help= 'Batch size', type= int, default= 1)
    parser.add_argument('-dataset', help= "path to modelnet point cloud data")
    parser.add_argument('-load', default= None)
    parser.add_argument('-save', default= None)
    parser.add_argument('-record', help= 'Record file name (e.g. record.txt)', default= None)
    parser.add_argument('-interval', type= int, help= 'Record interval within an epoch', default= 200)
    parser.add_argument('-support_num', type= int, default= 1)
    parser.add_argument('-neighbor_num', type= int, default= 20)
    parser.add_argument('-normal', dest= 'normal', action= 'store_true')
    parser.set_defaults(normal= False)
    parser.add_argument('-shift', type= float, default= None)
    parser.add_argument('-scale', type= float, default= None)
    parser.add_argument('-rotate', help="in degree", type= float, default= None)
    parser.add_argument('-axis', help= 'Rotation axis, select from [0, 1, 2]', type= int, default= 2) # upward axis = 2
    parser.add_argument('-random', help= 'Random transform in a given range', dest= 'random', action= 'store_true')
    parser.set_defaults(random= False)
    args = parser.parse_args()

    model = GCN3D(support_num= args.support_num, neighbor_num= args.neighbor_num)
    manager = Manager(model, args)

    transform = Transform(
        normal= args.normal,
        shift= args.shift,
        scale= args.scale,
        rotate= args.rotate,
        random= args.random
    )

    if args.mode == "train":
        print('Trianing ...')
        train_data = ModelNet_pointcloud(args.dataset, 'train', transform= transform)
        train_loader = DataLoader(train_data, shuffle= True, batch_size= args.bs)
        test_data = ModelNet_pointcloud(args.dataset, 'test', transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)
    
        manager.train(train_loader, test_loader)

    else:
        print('Testing ...')
        test_data = ModelNet_pointcloud(args.dataset, 'test', transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)
        
        test_loss, test_acc = manager.test(test_loader)

        print('Test Loss: {:.5f}'.format(test_loss))
        print('Test Acc:  {:.5f}'.format(test_acc))
        
if __name__ == '__main__':
    main()