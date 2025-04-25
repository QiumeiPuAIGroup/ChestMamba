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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.train_utils import train, plot_history
from utils.classify_4_dataset import COVIDDataset

import matplotlib.pyplot as plt

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
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training (default: 20)')
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
    parser.add_argument('--name', type=str, default='Mamba_classify_epoch30_4classify_draw2',
                        help='exp name for training')

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
    draw_confusion(y_true,y_pred)
    
def draw_confusion(y_true, y_pred, class_names = ['Normal', 'COVID', 'Viral Pneumonia', 'Lung Opacity']):
    cm = confusion_matrix(y_true, y_pred)
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    # 保存图片
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix3.png', dpi=300)  # 可设置 dpi 提高分辨率
    plt.close()  # 关闭图像以释放内存



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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000005, last_epoch=-1) # 动态更新学习率
    criterion = nn.CrossEntropyLoss()

    print_info()
    print('==> Training model..')
    Acc, Loss, Lr = train(model, trainloader=train_loader, testloader=val_loader, epochs=NUM_EPOCHS, optimizer=optimizer, scheduler=scheduler, criterion=criterion, path=save_path, verbose=True)
    plot_history(NUM_EPOCHS, Acc, Loss, Lr)

    eval('mamba_classify', False, save_dir=save_dir, test_loader=test_loader)
