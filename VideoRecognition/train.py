import timeit
from datetime import datetime
import socket
import os,sys
import glob,argparse
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model,X3D_model, slowfast_model,TAM_model
from video_swin_transformer import SwinTransformer3D
from timesformer.models.vit import TimeSformer
from braincog.model_zoo.vgg_snn import *
from braincog.utils import *
sys.path.append("/home/dongyiting/bullying10k/")  
from Bullying10k import *
# from ..bullying10k.Bullying10k import *
from sklearn.metrics import precision_recall_curve 
from sklearn.preprocessing import label_binarize
parser = argparse.ArgumentParser(description='c3d')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--step', type=int, default=16)
parser.add_argument('--gap', type=int, default=5)
parser.add_argument('--size', type=int, default=112)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--clip_grad', type=float, default=0)
parser.add_argument('--dataset', default='bullying10k', type=str) 
parser.add_argument('--model', default='C3D', type=str)
parser.add_argument('--opt', default='sgd', type=str)
args = parser.parse_args()


# Use GPU if available else revert to CPU
device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 1 # Run on test set every nTestInterval epochs
snapshot = 10 # Store a model every snapshot epochs
lr = args.lr # Learning rate

dataset = args.dataset # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
    args.channel=3
elif dataset == 'ucf101':
    num_classes = 101
    args.channel=3
elif dataset == 'bullying10k':
    num_classes = 10
    args.channel=2
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
 
if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = args.model # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False,inchannel=args.channel)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes,  layer_sizes=(2, 2, 2, 2),inchannel=args.channel )
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes,layer_sizes=(2, 2, 2, 2),  inchannel=args.channel)
        train_params = model.parameters()
    elif modelName == 'X3D':
        model = X3D_model.x3d_large(2,10)
        train_params = model.parameters()     
    elif modelName == 'TAM':
        model = TAM_model.TAM()
        train_params = model.parameters()     
    elif modelName == 'slowfast':
        model = slowfast_model.SlowFastmodel()
        train_params = model.parameters()        
                
    elif modelName == 'VGG_SNN':
        model = VGG_SNN(step=args.step,dataset=args.dataset,layer_by_layer=True)
        train_params = model.parameters()
    elif modelName == 'swintransformer':
        model = SwinTransformer3D(pretrained2d=False,in_chans=args.channel)
        train_params = model.parameters()
    elif modelName == 'timesformer':
        model = TimeSformer(img_size=224, num_classes=10,in_chans=2, num_frames=16, attention_type='divided_space_time' )
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    if args.opt=="sgd":optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt=="adamw":optimizer = optim.AdamW(train_params, lr=lr,weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    if dataset=="bullying10k":
        # train_dataloader = DataLoader( Bullying10k(r"/data/datasets/",train=True,step=16,gap=10) , batch_size=args.batch_size, shuffle=True, num_workers=4)
        # val_dataloader   = DataLoader(Bullying10k(r"/data/datasets/",train=False,step=16,gap=10) , batch_size=args.batch_size, num_workers=4)
        # test_dataloader  = DataLoader(Bullying10k(r"/data/datasets/",train=False,step=16,gap=10) , batch_size=args.batch_size, num_workers=4)
        train_dataloader,test_dataloader,_,_ = get_bullying10k_data(batch_size=args.batch_size,step=args.step,gap=args.gap,size=args.size)
        val_dataloader=train_dataloader
    else:
        train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=args.batch_size, num_workers=4)
        test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=args.batch_size, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device).float()
                if args.model!="VGG_SNN": 
                    inputs = inputs.transpose(-3,-4)
                labels = labels .to(device)
                optimizer.zero_grad()
 
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
 
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
 
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    if args.clip_grad>0:torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            yall=TensorGather()
            predall=TensorGather()
            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device).float()
                if args.model!="VGG_SNN": 
                    inputs = inputs.transpose(-3,-4)
                
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                yall.update(labels )
                predall.update(probs )
            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            
            y_true_bin = label_binarize(yall.gather.cpu().numpy(), classes=range(10))
            p=[]
            r=[]
            t=[]

            for i in range(10):

                precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], predall.gather[:,i].cpu().numpy())
                p.append(precision)       
                r.append(recall)       
                t.append(thresholds)       
 
            np.save("pr.npy",(p,r,t))
            writer.add_pr_curve(tag="pr_curve/val",labels=F.one_hot(yall.gather,10),predictions=F.softmax(predall.gather,dim=1),global_step=epoch)
            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()