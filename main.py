import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
from collections import defaultdict
import time
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



source_train_0_dir = hp.source_train_0_dir
source_train_1_dir = hp.source_train_1_dir


source_test_0_dir = hp.source_test_0_dir
source_test_1_dir = hp.source_test_1_dir




def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')  
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser



def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


    from data_function import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        # from models.two_d.alexnet import alexnet
        # model = alexnet(num_classes=2)

        # from models.two_d.densenet import densenet121
        # model = densenet121(num_class=2)

        from models.two_d.googlenet import googlenet
        model = googlenet(num_class=2)

        # from models.two_d.mobilenet import mobilenet
        # model = mobilenet(class_num=2)

        # from models.two_d.nasnet import nasnet
        # model = nasnet(class_num=2)

        # from models.two_d.resnet import resnet101
        # model = resnet101(num_classes=2)

        # from models.two_d.resnext import resnext101
        # model = resnext152(class_names=2)
        
        # from models.two_d.vggnet import vgg16_bn
        # model = vgg16_bn(num_class=2)

    elif hp.mode == '3d':

        from models.three_d.resnet3d import generate_model
        model = generate_model(18,n_input_channels=1,n_classes=2)

        from models.three_d.resnext3d import generate_model
        model = generate_model(50,n_input_channels=1,n_classes=2)

        from models.three_d.densenet3d import generate_model
        model = generate_model(121,n_input_channels=1,num_classes=2)








    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)


    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Classification_Loss
    criterion = Classification_Loss().cuda()


    writer = SummaryWriter(args.output_dir)



    train_dataset = MedData_train(source_train_0_dir,source_train_1_dir)
    train_loader = DataLoader(train_dataset.training_set, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)



    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

        num_iters = 0

        gts = []
        predicts = []
        for i, batch in enumerate(train_loader):
            

            if hp.debug:
                if i >=1:
                    break

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()


            x = batch['source']['data']
            y = batch['label']

            x = x.type(torch.FloatTensor).cuda()
            y = y.type(torch.LongTensor).cuda()

            if hp.mode == '2d':
                x = x.squeeze(-1)
                x = x[:,:1,:,:]

            outputs = model(x)

            outputs_logit = outputs.argmax(dim=1)

            loss = criterion(outputs, y, model)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1


            print("loss:"+str(loss.item()))
            writer.add_scalar('Training/Loss', loss.item(),iteration)

            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(y.cpu().detach().numpy())


        predicts = np.concatenate(predicts).flatten().astype(np.int16)
        gts = np.concatenate(gts).flatten().astype(np.int16)

        print(metrics.confusion_matrix(predicts, gts))
        acc = metrics.accuracy_score(predicts, gts) 
        recall = metrics.recall_score(predicts, gts) 
        f1 = metrics.f1_score(predicts, gts) 
        writer.add_scalar('Training/acc', acc,epoch)
        writer.add_scalar('Training/recall', recall,epoch)
        writer.add_scalar('Training/f1', f1,epoch)

        
        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )


        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {             
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )        

    writer.close()


def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test


    if hp.mode == '2d':
        # from models.two_d.alexnet import alexnet
        # model = alexnet(num_classes=2)

        # from models.two_d.densenet import densenet121
        # model = densenet121(num_class=2)

        from models.two_d.googlenet import googlenet
        model = googlenet(num_class=2)

        # from models.two_d.mobilenet import mobilenet
        # model = mobilenet(class_num=2)

        # from models.two_d.nasnet import nasnet
        # model = nasnet(class_num=2)

        # from models.two_d.resnet import resnet101
        # model = resnet101(num_classes=2)

        # from models.two_d.resnext import resnext101
        # model = resnext152(class_names=2)
        
        # from models.two_d.vggnet import vgg16_bn
        # model = vgg16_bn(num_class=2)

    elif hp.mode == '3d':

        from models.three_d.resnet3d import generate_model
        model = generate_model(18,n_input_channels=1,n_classes=2)

        from models.three_d.resnext3d import generate_model
        model = generate_model(50,n_input_channels=1,n_classes=2)

        from models.three_d.densenet3d import generate_model
        model = generate_model(121,n_input_channels=1,num_classes=2)




    model = torch.nn.DataParallel(model, device_ids=devicess,output_device=[1])


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    model.cuda()

    test_dataset = MedData_test(source_test_0_dir,source_test_1_dir)
    test_loader = DataLoader(test_dataset.testing_set, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.eval()

    predicts = []
    gts = []
    for i, batch in enumerate(test_loader):
        
        x = batch['source']['data']
        y = batch['label']

        x = x.type(torch.FloatTensor).cuda()
        y = y.type(torch.LongTensor).cuda()

        if hp.mode == '2d':
            x = x.squeeze(-1)
            x = x[:,:1,:,:]

        outputs = model(x)

        outputs_logit = outputs.argmax(dim=1)

        predicts.append(outputs_logit.cpu().detach().numpy())
        gts.append(y.cpu().detach().numpy())

    predicts = np.concatenate(predicts).flatten().astype(np.int16)
    gts = np.concatenate(gts).flatten().astype(np.int16)
    acc = metrics.accuracy_score(predicts, gts) 
    recall = metrics.recall_score(predicts, gts) 
    f1 = metrics.f1_score(predicts, gts) 
    ## log
    print("acc:"+str(acc))
    print("recall:"+str(recall))
    print("f1:"+str(f1))
    print(metrics.confusion_matrix(predicts, gts))




         


   

if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
