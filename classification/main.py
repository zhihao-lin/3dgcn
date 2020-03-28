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
    parser.add_argument('-mode', default= 'train', help= '[train/test]')
    parser.add_argument('-model', default= 'gcn3d', help= '[pointnet/dgcnn/gcn3d]')
    parser.add_argument('-cuda', default= '0', help= 'Cuda index')
    parser.add_argument('-epoch', type= int, default= 100, help= 'Epoch number')
    parser.add_argument('-lr', type= float, default= 1e-4, help= 'Learning rate')
    parser.add_argument('-bs', type= int, default= 1, help= 'Batch size')
    parser.add_argument('-dataset', help= "Path to modelnet point cloud data")
    parser.add_argument('-load', help= 'Path to load model')
    parser.add_argument('-save', help= 'Path to save model')
    parser.add_argument('-record', help= 'Record file name (e.g. record.log)')
    parser.add_argument('-interval', type= int, default= 200, help= 'Record interval within an epoch')
    parser.add_argument('-support', type= int, default= 1, help= 'Support number')
    parser.add_argument('-neighbor', type= int, default= 20, help= 'Neighbor number')
    parser.add_argument('-normal', dest= 'normal', action= 'store_true', help= 'Normalize objects (zero-mean, unit size)')
    parser.set_defaults(normal= False)
    parser.add_argument('-shift', type= float, help= 'Shift objects (original: 0.0)')
    parser.add_argument('-scale', type= float, help= 'Enlarge/shrink objects (original: 1.0)')
    parser.add_argument('-rotate', type= float, help= 'Rotate objects in degree (original: 0.0)')
    parser.add_argument('-axis', type= int, default= 2, help= 'Rotation axis [0, 1, 2] (upward = 2)')
    parser.add_argument('-random', dest= 'random', action= 'store_true', help= 'Randomly transform in a given range')
    parser.set_defaults(random= False)
    args = parser.parse_args()

    model = GCN3D(support_num= args.support, neighbor_num= args.neighbor)
    if args.model == 'pointnet':
        model = PointNetCls(40)
    elif args.model == 'dgcnn':
        model = DGCNN()

    manager = Manager(model, args)

    transform = Transform(
        normal= args.normal,
        shift= args.shift,
        scale= args.scale,
        rotate= args.rotate,
        axis= args.axis,
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