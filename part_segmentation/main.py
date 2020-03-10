import sys
sys.path.append('..')
from uitl import Transform
import argparse
import torch
from torch.utils.data import DataLoader
from model_gcn3d import GCN3D
from shapenet import ShapeNetPart
from manager import Manager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", help= "train/test", default= "train")
    parser.add_argument('-cuda', help= "Cuda index", default= "0")
    parser.add_argument('-epoch', type= int, default= 20)
    parser.add_argument('-lr', type= float, default= 1e-4)
    parser.add_argument('-bs', help= 'Batch size', type= int, default= 1)
    parser.add_argument('-dataset', help= "path to ShapeNetPart")
    parser.add_argument('-load', default= None)
    parser.add_argument('-save', default= None)
    parser.add_argument('-record', help= 'Record file name (e.g. record.log)', default= None)
    parser.add_argument('-interval', type= int, help= 'Record interval within an epoch', default= None)
    parser.add_argument('-point_num', type= int, default= 1024)
    parser.add_argument('-support_num', type= int, default= 1)
    parser.add_argument('-neighbor_num', type= int, default= 20)
    parser.add_argument('-output', help= 'folder for visualization images', default= None)
    parser.add_argument('-normal', dest= 'normal', action= 'store_true')
    parser.set_defaults(normal= False)
    parser.add_argument('-shift', type= float, default= None)
    parser.add_argument('-scale', type= float, default= None)
    parser.add_argument('-rotate', help="in degree", type= float, default= None)
    parser.add_argument('-axis', help= 'Rotation axis, select from [0, 1, 2]', type= int, default= 2) # upward axis = 2
    parser.add_argument('-random', help= 'Random transform in a given range', dest= 'random', action= 'store_true')
    parser.set_defaults(random= False)
    args = parser.parse_args()

    model = GCN3D(class_num= 50, support_num= args.support_num, neighbor_num= args.neighbor_num)
    manager = Manager(model, args)

    transform = Transform(
        normal= args.normal,
        shift= args.shift,
        scale= args.scale,
        rotate= args.rotate,
        random= args.random
    )

    if args.mode == "train":
        print("Training ...")
        train_data = ShapeNetPart(args.dataset, class_choice= None, split= 'train', point_num= args.point_num, transform= transform)
        train_loader = DataLoader(train_data, shuffle= True, batch_size= args.bs)
        test_data = ShapeNetPart(args.dataset, class_choice= None, split= 'test', point_num= args.point_num, transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)

        manager.train(train_loader, test_loader)

    elif args.mode == "test":
        print("Testing ...")
        test_data = ShapeNetPart(args.dataset, class_choice= None, split= 'test', point_num= args.point_num, transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)

        test_loss, test_table_str = manager.test(test_loader, args.output)
        print(test_table_str)
        
if __name__ == '__main__':
    main()