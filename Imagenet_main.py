import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import My_BatchSampler




model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='opt', help='sgd/rmsprop/adam')

parser.add_argument('--print_freq', default=10, type=int,
                    metavar='print_freq', help='print freq')

parser.add_argument('--sample_mode', default='with_replacement', type=str,
                    metavar='sample_mode', help='with_replacement (others don\'t work for the moment)')


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--rho', default=0.9, type=float,
                    metavar='rho', help='RMSprop rho')
parser.add_argument('--epsilon', default=1e-30, type=float,
                    metavar='epsilon', help='RMSprop epsilon')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume_ckpt', action='store_true',
                    help='whether to start from previous checkpoint')
                    
                    #default='', type=str, metavar='PATH',
                    #help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--warmup', default=0, type=int,
                    metavar='warmup', help='Wanna have a warmup: 0/1')
parser.add_argument('--warmup_steps', default=5000, type=int,
                    metavar='warmupsteps', help='How many warmup steps you want? (5k default)')


parser.add_argument('--schedule_lr', default=0, type=int,
                    metavar='schedule_lr', help='0/1')

parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='gamma', help='Rate of lr annealing: 0.1 (default)')

parser.add_argument('--schedule_pattern', default='60-75-90', type=str,
                    metavar='schedule_lr_pattern', help='60-75-90')

parser.add_argument('--drop_last', default=1, type=int,
                    metavar='drop_last', help='0/1 (drop last batch)')

parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='Imagenet_checkpoints', type=str)

parser.add_argument('--wandb_project', 
                    help='The wandb project to upload your results',
                    default='', type=str)


parser.add_argument('--wandb_entity', 
                    help='The wandb entity to upload your results',
                    default='', type=str)

best_acc1 = 0




    
def main():
    args = parser.parse_args()
    
    
    
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    config = {}
    config['batch_size'] = args.batch_size 
    config['arch'] = args.arch   
    config['effective lr'] = args.lr 
    
    config['schedule_pattern'] = args.schedule_pattern
    config['optimizer'] = args.optimizer
    
    if args.optimizer == 'rmsprop':
        config['rho'] = args.rho 
        config['momentum'] = 0.0
        config['epsilon'] = args.epsilon
    
    if args.optimizer == 'adam':
        config['rho'] = args.rho 
        config['momentum'] = args.momentum
        config['epsilon'] = args.epsilon
    
    if args.optimizer == 'sgd':
        config['rho'] = 0. 
        config['momentum'] = 0.
        config['epsilon'] = 0.
         
    sub_dir = '_'.join([str(key)+ '_' + str(config[key]) for key in ['arch', 'batch_size', 'effective lr', 'rho', 'momentum', 'epsilon', 'optimizer', 'schedule_pattern']])
    
    args.save_dir += '/' + sub_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    if args.resume_ckpt:    
        args.resume = args.save_dir + '/checkpoint.pth.tar'
    else:
        args.resume = None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

        
        

def main_worker(gpu, ngpus_per_node, args):
    
    global best_acc1
    args.gpu = gpu

    
    config = {}
    config['batch_size'] = args.batch_size 
    
    #if args.arch != 'vgg_manual':
    config['arch'] = args.arch
    #else:
    #    config['arch'] = args.arch + '_' + args.vgg_cfg + '_' + str(args.bn) + '_' + str(args.act)
        
        
    #if args.train_type == 0:
    #    config['effective lr'] = args.lr * ( args.data_subset/args.batch_size ) 
    #else:
    #    config['effective lr'] = args.lr 
   
    config['effective lr'] = args.lr 
    
        
    config['optimizer'] = args.optimizer
    
    if args.optimizer == 'rmsprop':
        config['rho'] = args.rho 
        config['momentum'] = 0.0
        config['epsilon'] = args.epsilon
    
    if args.optimizer == 'adam':
        config['rho'] = args.rho 
        config['momentum'] = args.momentum
        config['epsilon'] = args.epsilon
    
    if args.optimizer == 'sgd':
        config['rho'] = 0. 
        config['momentum'] = 0.
        config['epsilon'] = 0.
        
    config['schedule_lr'] = args.schedule_lr
    config['schedule_pattern'] = args.schedule_pattern
    config['weight_decay'] = args.weight_decay
    config['sample_mode'] = args.sample_mode
    
    config['warmup'] = args.warmup
    
    config['epochs'] = args.epochs 
    config['warmup_steps'] = args.warmup_steps
    
    
    config['manual_seed'] = args.seed
    
    config['gamma'] = args.gamma
   
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, group='Model ' + ' '.join([str(key)+ ' ' + str(config[key]) for key in ['batch_size', 'effective lr', 'rho', 'momentum', 'epsilon'] ]), settings=wandb.Settings(start_method='fork') )
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            args.ngpus_per_node = ngpus_per_node
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    if args.optimizer == 'sgd':    
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop' :    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.0, args.rho), eps=args.epsilon, weight_decay=args.weight_decay, amsgrad=False)
          
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.rho), eps=args.epsilon, weight_decay=args.weight_decay, amsgrad=False)
        
    
    initial_progress = 0 
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            #try:
            best_acc1 = checkpoint['best_acc1']
            #except:
            initial_progress = checkpoint['progress']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    #if args.distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #else:
    #    train_sampler = None
    dataset_size = 1281167
    looper = max(args.batch_size // 256, 1)
    train_sampler = My_BatchSampler(dataset_size=dataset_size, batch_size=min(args.batch_size, 256), drop_last=args.drop_last, sample_mode = args.sample_mode, num_replicas=args.world_size)
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        num_workers=args.workers, pin_memory=True, batch_sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, 0, wandb)
        return

    if args.schedule_lr == 1:
        schedule_pattern = [int(k) for k in args.schedule_pattern.split('-')] 
        
        
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)
        if args.schedule_lr == 1:  
            adjust_learning_rate(optimizer, epoch, schedule_pattern, args.gamma)
            
            
        if args.warmup == 1:
            if epoch <= args.warmup_steps:
                effective_lr = args.lr
                warmup_learning_rate(optimizer, epoch, effective_lr * 1e-3, effective_lr, args.warmup_steps)
        
        

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, looper, wandb, best_acc1, initial_progress)
        initial_progress = 0

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, wandb)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'progress': 0,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            
            if epoch % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'progress': 0,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint-'+ str(epoch) + '.pth.tar'))
            
        
    wandb.finish()

def train(train_loader, model, criterion, optimizer, epoch, args, looper, wandb, best_acc1, initial_progress):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    count_looper = 0

    end = time.time()
    
    
    for i, (images, target) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)
        count_looper += 1

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        if count_looper == 1:
            optimizer.zero_grad()
        #else:
        #    loss += criterion(output, target)
        loss /= looper
        
        loss.backward()        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #if count_looper == looper:
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if count_looper == looper:
            optimizer.step()
            count_looper = 0    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i % 1000 == 0 and i > 0:    
            param_norm = 0.
            for p in model.parameters():
                param_norm += torch.linalg.norm(p).float() ** 2

            grad_norm = 0.
            for p in model.parameters():
                grad_norm += torch.linalg.norm(p.grad).float() ** 2
                

            wandb.log({'Param norm': param_norm ** 0.5, "epoch": (epoch - 1) + (i / (1. * len(train_loader)) ) })
            wandb.log({'Grad norm': grad_norm ** 0.5, "epoch": (epoch - 1) + (i / (1. * len(train_loader)) ) })
            wandb.log({'train loss': losses.avg, "epoch": (epoch - 1) + (i / (1. * len(train_loader)) ) })
            wandb.log({'train prec @1': top1.avg, "epoch": (epoch - 1) + (i / (1. * len(train_loader)) ) })
            wandb.log({'train prec @5': top5.avg, "epoch": (epoch - 1) + (i / (1. * len(train_loader)) ) })
        
        
        if i % 100 == 0: 
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % args.ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch,
                    'progress': i,
                    'arch': args.arch,
                    'best_acc1': best_acc1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, False, filename=os.path.join(args.save_dir, 'checkpoint.pth.tar'))    
         
        if initial_progress != 0 and i >= len(train_loader) - initial_progress:
            break
            

    #wandb.log({'Param norm': param_norm ** 0.5, "epoch": epoch })
    #wandb.log({'Grad norm': grad_norm ** 0.5, "epoch": epoch })
    wandb.log({'train loss': losses.avg, "epoch": epoch })
    wandb.log({'train prec @1': top1.avg, "epoch": epoch })
    wandb.log({'train prec @5': top5.avg, "epoch": epoch })

def validate(val_loader, model, criterion, args, epoch, wandb):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
    
    
    wandb.log({'test loss': losses.avg, "epoch": epoch })
    wandb.log({'test prec @1': top1.avg, "epoch": epoch })
    wandb.log({'test prec @5': top5.avg, "epoch": epoch })
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


    
def warmup_learning_rate(optimizer, epoch, init_lr, max_lr, total_warmup_steps):
    curr_lr = init_lr * ((max_lr / init_lr) ** (epoch / total_warmup_steps)) #init_lr + epoch * (max_lr - init_lr) / total_warmup_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = curr_lr
    
def adjust_learning_rate(optimizer, epoch, schedule_pattern, gamma):
    curr_eff_epoch = epoch  
    # // int(data_size / batch_size)
    if curr_eff_epoch in schedule_pattern:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
        schedule_pattern.pop(0)
            

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()