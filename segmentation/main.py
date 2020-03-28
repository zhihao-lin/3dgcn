import sys
sys.path.append('..')
from util import Transform
import argparse
import torch
from torch.utils.data import DataLoader
from model_gcn3d import GCN3D
from dataset_shapenet import ShapeNetPart
from manager import Manager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default= 'train', help= '[train/test]')
    parser.add_argument('-cuda', default= '0', help= 'Cuda index')
    parser.add_argument('-epoch', type= int, default= 50, help= 'Epoch number')
    parser.add_argument('-lr', type= float, default= 1e-4, help= 'Learning rate')
    parser.add_argument('-bs', type= int, default= 1, help= 'Batch size')
    parser.add_argument('-dataset', help= "Path to ShapeNetPart")
    parser.add_argument('-load', help= 'Path to load model')
    parser.add_argument('-save', help= 'Path to save model')
    parser.add_argument('-record', help= 'Record file name (e.g. record.log)')
    parser.add_argument('-interval', type= int, help= 'Record interval within an epoch')
    parser.add_argument('-point', type= int, default= 1024, help= 'Point number per object')
    parser.add_argument('-support', type= int, default= 1, help= 'Support number')
    parser.add_argument('-neighbor', type= int, default= 50, help= 'Neighbor number')
    parser.add_argument('-output', help= 'Folder for visualization images')
    parser.add_argument('-normal', dest= 'normal', action= 'store_true', help= 'Normalize objects (zero-mean, unit size)')
    parser.set_defaults(normal= False)
    parser.add_argument('-shift', type= float, help= 'Shift objects (original: 0.0)')
    parser.add_argument('-scale', type= float, help= 'Enlarge/shrink objects (original: 1.0)')
    parser.add_argument('-rotate', type= float, help= 'Rotate objects in degree (original: 0.0)')
    parser.add_argument('-axis', type= int, default= 1, help= 'Rotation axis [0, 1, 2] (upward = 1)') # upward axis = 1
    parser.add_argument('-random', dest= 'random', action= 'store_true', help= 'Randomly transform in a given range')
    parser.set_defaults(random= False)
    args = parser.parse_args()

    model = GCN3D(class_num= 50, support_num= args.support, neighbor_num= args.neighbor)
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
        print("Training ...")
        train_data = ShapeNetPart(args.dataset, split= 'train', point_num= args.point, transform= transform)
        train_loader = DataLoader(train_data, shuffle= True, batch_size= args.bs)
        test_data = ShapeNetPart(args.dataset, split= 'test', point_num= args.point, transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)

        manager.train(train_loader, test_loader)

    elif args.mode == "test":
        print("Testing ...")
        test_data = ShapeNetPart(args.dataset, split= 'test', point_num= args.point, transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)

        test_loss, test_table_str = manager.test(test_loader, args.output)
        print(test_table_str)
        
if __name__ == '__main__':
    main()