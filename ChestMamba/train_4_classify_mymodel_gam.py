# IMPORT PACKAGES
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import argparse
import random
import yaml
import os

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import random_split

from utils.classifier_dataset import PneumoniaDataset, ImageTransform
from utils.common_utils import create_folder, print_info

from models.create_model import create_model

from metrics_learn import classification_metrics
from utils.train_utils import train, plot_history
from utils.classify_4_dataset import COVIDDataset

from gam.gam import GAM
from gam.smooth_cross_entropy import smooth_crossentropy
from gam.util import ProportionScheduler

# from onnx.export_onnx import export_onnx
# from compression.prune import main_prune

# result reproduction
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # Train Options
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training (default: 32)')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs for training (default: 20)')
    parser.add_argument('--optim-policy', type=str, default='adam', help='optimizer for training. [sgd | adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight-decay for training. [le-4 | 1e-6]')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd (default: 0.9)')
    parser.add_argument('--lr-policy', type=str, default='cos', help='learning rate decay policy. [step | cos]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiply by a gamma every lr_decay_epochs (only for step lr policy)')
    parser.add_argument('--step-size', type=int, default=20, help='period of lr decay. (default: 20)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed. [47 | 3407 | 1234]')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--pretrain', action='store_true', default=True, help='whether to fine-tune')

    # Model Options
    parser.add_argument('--model_name', type=str, default='mamba_classify',
                        help='CNN [resnet18-cbam | resnet18 | seresnet18 | resnet34 | vgg16 | vgg19 | mobilenetv2 | alexnet]')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='whether to fine-tune on pretrained model')

    '''
    # Prune Options
    parser.add_argument('--baseline', type=str,
                        default='checkpoints/cls/exp_fake_scratch_pretrained_test/pneumonia_best.pth',
                        help='base model for pruning')
    parser.add_argument('--prune', action='store_true', default=False, help='start pruning')
    parser.add_argument('--prune-policy', type=str, default='fpgm', help='pruning policy. [l1 | l2 | fpgm] ')
    parser.add_argument('--sparse', type=float, default=0.8, help='pruning ratio (default: 0.8)')
    '''

    # File Management Options
    parser.add_argument('--name', type=str, default='Mamba_classify_epoch30_classify_gam_withpretrained_lr0.0001_2292',
                        help='exp name for training')
    
    parser.add_argument("--grad_beta_0", default=0.1, type=float, help="scale for g0")
    parser.add_argument("--grad_beta_1", default=0.1, type=float, help="scale for g1")
    parser.add_argument("--grad_beta_2", default=0.9, type=float, help="scale for g2")
    parser.add_argument("--grad_beta_3", default=0.9, type=float, help="scale for g3")
    parser.add_argument("--grad_gamma", default=0.03, type=int, help="")

    return parser

def eval(model_name, use_pretrained, save_dir, test_loader):
    model = create_model(model_name, use_pretrained).to('cuda')
    ckpt_path = f'{save_dir}/best.pth'
    model = torch.load(ckpt_path)
    # model.load_state_dict(state_dict)

    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            print(outputs.shape)

            _, pred = torch.max(outputs, 1)
            print(pred.shape)
            y_true += targets.tolist()
            print(targets.tolist())
            y_pred += pred.tolist()
            print(pred.tolist())

    classification_metrics(Y_test=y_true, Y_pred=y_pred, n=4)

def train_model(net, trainloader, testloader, epochs, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets, smoothing=0.1).mean()
    def get_acc(outputs, label):
        total = outputs.shape[0]
        probs, pred_y = outputs.data.max(dim=1) # 得到概率
        correct = (pred_y == label).sum().data
        return torch.div(correct, total)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    lr_list  = []
    for i in range(epochs):
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        train_step = len(trainloader)
        with tqdm(total=train_step,desc=f'Train Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,data in enumerate(trainloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # formard
                optimizer.set_closure(loss_fn, im, label)
                outputs, loss = optimizer.step()
                # 累计损失
                train_loss += loss.item()
                train_acc += get_acc(outputs,label).item()
                pbar.set_postfix(**{'Train Acc' : train_acc/(step+1),
                                'Train Loss' :train_loss/(step+1)})
                pbar.update(1)
            pbar.close()
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # 更新学习率
        scheduler.step(train_loss)
        if testloader is not None:
            net.eval()
            test_step = len(testloader)
            with torch.no_grad():
                with tqdm(total=test_step,desc=f'Test Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
                    for step,data in enumerate(testloader,start=0):
                        im,label = data
                        im = im.to(device)
                        label = label.to(device)
                        # 释放内存
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        outputs = net(im)
                        loss = loss_fn(outputs,label)
                        test_loss += loss.item()
                        test_acc += get_acc(outputs,label).item()
                        pbar.set_postfix(**{'Test Acc' : test_acc/(step+1),
                                'Test Loss' :test_loss/(step+1)})
                        pbar.update(1)
                    pbar.close()
                test_loss = test_loss / len(testloader)
                test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            print(
                'Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epochs, train_loss, train_acc, test_loss, test_acc,lr))
        else:
            print('Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epochs,train_loss,train_acc,lr))

        # ====================== 使用 tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'test': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'test': test_acc}, i+1)
            writer.add_scalar('Learning Rate',lr,i+1)
        # =========================================================
        # 如果取得更好的准确率，就保存模型
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr


if __name__ == '__main__':
    args = get_argparse().parse_args()

    # hyper-parameters
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    LR = args.lr
    MOMENTUM = args.momentum
    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # data info
    with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset_root1 = yaml_info['root1']
    dataset_root2 = yaml_info['root2']
    n_channels = yaml_info['n_channels']
    n_classes = yaml_info['n_classes']
    image_size = yaml_info['image_size']

    # Set save dir
    train_infos = [args.name, 'lr', LR, args.lr_policy, args.step_size]
    log_info = "_".join(str(i) for i in train_infos)
    save_dir = f"checkpoints/cls/{log_info}"
    create_folder(save_dir)
    save_path = os.path.join(save_dir, 'best.pth')
    # print(f"save_dir={save_dir}")

    # Create the table object, name, and alignment
    table = PrettyTable(['Hyper-Parameters & data infos', 'Value'])
    table.align['Hyper-Parameters & data infos'] = 'l'
    table.align['Value'] = 'r'

    # Add to table
    table.add_row(['Batch size', BATCH_SIZE])
    table.add_row(['Workers', WORKERS])
    table.add_row(['Num epochs', NUM_EPOCHS])
    table.add_row(['Optimizer strategy', args.optim_policy])
    table.add_row(['Weight decay', args.weight_decay])
    table.add_row(['Learning rate', LR])
    table.add_row(['Momentum', MOMENTUM])
    table.add_row(['LR policy', args.lr_policy])
    table.add_row(['gamma', args.gamma])
    table.add_row(['step-size', args.step_size])
    table.add_row(['random seed', args.seed])
    table.add_row(['Device', DEVICE])
    table.add_row(["", ""])
    table.add_row(['dataset_root1', dataset_root1])
    table.add_row(['dataset_root2', dataset_root2])
    table.add_row(['n_channels', n_channels])
    table.add_row(['n_classes', n_classes])
    table.add_row(['image_size', image_size])
    print(table)

    set_seed(args.seed)

    # Get dataset and prepare dataloader
    print_info()
    print('==> Getting dataloader..')
    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256,256]),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = COVIDDataset(root='data/COVID-19_Radiography_Dataset', transform=transform_)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size -val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(49))
    print_info()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Create models
    print_info()
    print('==> Building model..')
    model = create_model(args.model_name, args.use_pretrained).to(DEVICE)
    model.load_from()

    print_info()
    print('==> Defining optimizer and scheduler..')

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # 动态更新学习率
    criterion = nn.CrossEntropyLoss()

    grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.0001, min_lr=0.0,
                                                 max_value=0.00002, min_value=0.00002)

    grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.0001, min_lr=0.0,
                                                      max_value=0.0002,
                                                      min_value=0.0002)
    
    gam_optimizer = GAM(params=model.parameters(), base_optimizer=optimizer, model=model, grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler, args=args)

    print_info()
    print('==> Training model..')
    Acc, Loss, Lr = train_model(model, trainloader=train_loader, testloader=val_loader, epochs=NUM_EPOCHS, optimizer=gam_optimizer, scheduler=scheduler, criterion=None, path=save_path, verbose=True)
    plot_history(NUM_EPOCHS, Acc, Loss, Lr, save_dir=save_dir)

    eval('mamba_classify', False, save_dir=save_dir, test_loader=test_loader)
