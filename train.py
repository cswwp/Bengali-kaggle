import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
# from resnet import ResNet34
from data import GraphemeDataset
from metric import macro_recall_multi, macro_recall_multi_mixup
from data import generate_data_loader
from NET.efficientnet import EfficientNet
import pandas as pd
import gc
import torch.optim
import argparse
from lr_cos_restart import CosineAnnealingLR_with_Restart
from logger import Logger
from mixup import cutmix, mixup, mixup_criterion, mixup_criterion_with_ohem, mixup_criterion_with_focal_loss
from radam import RAdam, AdamW

def get_args():
    parser = argparse.ArgumentParser(description="Train program for BELI.")
    parser.add_argument('--model', type=str, default='efficientnet-b4')
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM', 'RADAM', 'ADAMW', 'OVER9000'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--outdir', type=str, default='models')
    parser.add_argument('--gpu_ids', type=str, default='3,4')
    parser.add_argument('--log_interval', type=int, default=2000)
    # parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--csv_path', type=str, default='BengaliData')
    parser.add_argument('--feather_data_path', type=str, default='BengaliData/feather128')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--cycle_inter', type=int, default=200)
    parser.add_argument('--cycle_num', type=int, default=1)
    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--mixup', type=int, default=0)

    parser.add_argument('--alpha_cutmix', type=float, default=0.5)
    parser.add_argument('--alpha_mixup', type=float, default=0.2)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--finetune', default=1, type=int)
    parser.add_argument('--image_mode', type=str, default='gray')
    parser.add_argument('--LR_SCHEDULER', type=str, default='COS', choices=['REDUCED', 'COS'])
    parser.add_argument('--lr_ratio', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default='2')

    # parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    print(args)

    return args


## This function for train is copied from @hanjoonchoe
## We are going to train and track accuracy and then evaluate and track validation accuracy
def train(epoch, train_loader, model, optimizer, criterion, log, args):
    model.train()
    losses = []
    accs = []
    acc = 0.0
    total = 0.0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0

    recall_grapheme_all = 0.0
    recall_vowel_all = 0.0
    recall_consonant_all = 0.0
    #ohem_percent = .6
    # print('straing')
    first_run = True
    for idx, (inputs, labels1, labels2, labels3) in enumerate(tqdm(train_loader)):
        if first_run:
            plt.imshow()
            
        #print('input()', inputs.shape, labels1.shape, labels2.shape, labels3.shape)
        if args.mixup:
            if 0:#np.random.rand()<0.5:
                inputs, labels1, labels2, labels3 = mixup(inputs, labels1, labels2, labels3, args.alpha_mixup)
            else:
                inputs, labels1, labels2, labels3 = cutmix(inputs, labels1, labels2, labels3, args.alpha_cutmix)
            inputs = inputs.cuda()  # to(device)

            t1_0, t1_1, t1_2 = labels1  # [0], labels1[1], labels1[2]
            t2_0, t2_1, t2_2 = labels2  # [0], labels2[1], labels2[2]
            t3_0, t3_1, t3_2 = labels3  # [0], labels3[1], labels3[2]

            labels1 = (t1_0.cuda(), t1_1.cuda(), t1_2)
            labels2 = (t2_0.cuda(), t2_1.cuda(), t2_2)
            labels3 = (t3_0.cuda(), t3_1.cuda(), t3_2)
        else:
            inputs = inputs.cuda()  # to(device)
            labels1 = labels1.cuda()  # to(device)
            labels2 = labels2.cuda()  # to(device)
            labels3 = labels3.cuda()  # to(device)

        total += len(inputs)
        optimizer.zero_grad()
        outputs1, outputs2, outputs3 = model(inputs.float())

        if args.mixup:
            loss1 = mixup_criterion(outputs1, labels1)
            #loss1, _ = loss1.topk(k=int(ohem_percent * args.batch_size))

            loss2 = mixup_criterion(outputs2, labels2) * 2

            # focal loss
            #loss2 = mixup_criterion_with_focal_loss(outputs2, labels2)

            # ohem loss
            # if epoch<= 100:
            #     ohem_percent = 0.7
            # elif epoch>100 and epoch <= 200:
            #     ohem_percent = 0.5
            # else:
            #     ohem_percent = 0.3
            #
            # loss2 = mixup_criterion_with_ohem(outputs2, labels2)
            # loss2, _ = loss2.topk(k=int(ohem_percent * loss2.shape[0]), dim=0)
            # loss2 = loss2.mean()*2


            loss3 = mixup_criterion(outputs3, labels3)
            #loss3, _ = loss3.topk(k=int(ohem_percent * args.batch_size))

        else:
            loss1 = criterion(outputs1, labels1)
            loss2 = 2 * criterion(outputs2, labels2)
            loss3 = criterion(outputs3, labels3)

        running_loss += loss1.item() + loss2.item() + loss3.item()
        if args.mixup:
            targets1_1, targets1_2, alpha1 = labels1
            targets2_1, targets2_2, alpha2 = labels2
            targets3_1, targets3_2, alpha3 = labels3

            average_recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi_mixup(outputs2, targets2_1, targets2_2, alpha2,
                                                       outputs1, targets1_1, targets1_2, alpha1,
                                                       outputs3, targets3_1, targets3_2, alpha3)

            running_recall += average_recall

            recall_grapheme_all += recall_grapheme
            recall_vowel_all += recall_vowel
            recall_consonant_all += recall_consonant

            running_acc += alpha1 * (outputs1.argmax(1) == targets1_1).float().mean() + \
                           (1 - alpha1) * (outputs1.argmax(1) == targets1_2).float().mean()
            running_acc += alpha2 * (outputs2.argmax(1) == targets2_1).float().mean() + \
                           (1 - alpha2) * (outputs2.argmax(1) == targets2_2).float().mean()
            running_acc += alpha3 * (outputs3.argmax(1) == targets3_1).float().mean() + \
                           (1 - alpha3) * (outputs3.argmax(1) == targets3_2).float().mean()


        else:
            average_recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi(outputs2, labels2, outputs1, labels1, outputs3, labels3)
            running_recall += average_recall
            recall_grapheme_all += recall_grapheme
            recall_vowel_all += recall_vowel
            recall_consonant_all += recall_consonant
            running_acc += (outputs1.argmax(1) == labels1).float().mean()
            running_acc += (outputs2.argmax(1) == labels2).float().mean()
            running_acc += (outputs3.argmax(1) == labels3).float().mean()

        (loss1 + loss2 + loss3).backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = running_acc / total
        # scheduler.step()
    losses.append(running_loss / len(train_loader))
    accs.append(running_acc / (len(train_loader) * 3))
    log.write(' train epoch : {}\tacc : {:.2f}%\n'.format(epoch, running_acc / (len(train_loader) * 3)))
    log.write('loss : {:.4f}\n'.format(running_loss / len(train_loader)))

    log.write('recall_grapheme: {}\t recall_vowel: {}\t recall_consonant: {}\n'.format(recall_grapheme_all/len(train_loader),
              recall_vowel_all/len(train_loader), recall_consonant_all/len(train_loader)))

    log.write('recall: {}\n'.format(running_recall / len(train_loader)))
    total_train_recall = running_recall / len(train_loader)
    torch.cuda.empty_cache()
    gc.collect()
    # history.loc[epoch, 'train_loss'] = losses[0]
    # history.loc[epoch, 'train_acc'] = accs[0].cpu().numpy()
    # history.loc[epoch, 'train_recall'] = total_train_recall
    return total_train_recall


def evaluate(epoch, model, criterion, valid_loader, log):
    model.eval()
    losses = []
    accs = []
    recalls = []
    acc = 0.0
    total = 0.0
    # print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0

    recall_grapheme_all = 0.0
    recall_vowel_all = 0.0
    recall_consonant_all = 0.0

    with torch.no_grad():
        for idx, (inputs, labels1, labels2, labels3) in enumerate(tqdm(valid_loader)):
            inputs = inputs.cuda()  # to(device)
            labels1 = labels1.cuda()  # to(device)
            labels2 = labels2.cuda()  # to(device)
            labels3 = labels3.cuda()  # to(device)
            total += len(inputs)
            outputs1, outputs2, outputs3 = model(inputs.float())
            loss1 = criterion(outputs1, labels1)
            #focal loss for root cls
            #loss2 = mixup_criterion_with_focal_loss(outputs2, labels2)
            loss2 = 2*criterion(outputs2, labels2)
            loss3 = criterion(outputs3, labels3)
            running_loss += loss1.item() + loss2.item() + loss3.item()

            # running_recall += macro_recall_multi(outputs2, labels2, outputs1, labels1, outputs3, labels3)
            # running_acc += (outputs1.argmax(1) == labels1).float().mean()
            # running_acc += (outputs2.argmax(1) == labels2).float().mean()
            # running_acc += (outputs3.argmax(1) == labels3).float().mean()
            # acc = running_acc / total
            average_recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi(outputs2, labels2, outputs1, labels1, outputs3, labels3)
            running_recall += average_recall
            recall_grapheme_all += recall_grapheme
            recall_vowel_all += recall_vowel
            recall_consonant_all += recall_consonant
            running_acc += (outputs1.argmax(1) == labels1).float().mean()
            running_acc += (outputs2.argmax(1) == labels2).float().mean()
            running_acc += (outputs3.argmax(1) == labels3).float().mean()
            acc = running_acc / total

            # scheduler.step()
    losses.append(running_loss / len(valid_loader))
    accs.append(running_acc / (len(valid_loader) * 3))
    recalls.append(running_recall / len(valid_loader))
    total_recall = running_recall / len(valid_loader)  ## No its not Arnold Schwarzenegger movie
    log.write('val epoch: {} \tval acc : {:.2f}%\n'.format(epoch, running_acc / (len(valid_loader) * 3)))
    log.write('loss : {:.4f}\n'.format(running_loss / len(valid_loader)))

    log.write('recall_grapheme:{}\trecall_vowel:{}\trecall_consonant:{}\n'.format(recall_grapheme_all/len(valid_loader),
              recall_vowel_all/len(valid_loader), recall_consonant_all/len(valid_loader)))

    log.write('recall: {}\n'.format(running_recall / len(valid_loader)))

    return total_recall


def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    from opt.ralamb import Ralamb
    from opt.lookahead import Lookahead
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    log = Logger()
    log.open(os.path.join(args.outdir, args.model + '_log.txt'), mode='a')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    log.write('TRAINING BENGALI\n')
    log.write('BATCHS SIZE:%d\n' % args.batch_size)
    log.write('OUT DIR:%s\n' % args.outdir)
    log.write('MODEL:%s\n' % args.model)
    log.write('OPTIMIZER:%s\n' % args.optimizer)
    log.write('CYCLE INTER:%d\n' % args.cycle_inter)
    log.write('CYCLE NUM:%d\n' % args.cycle_num)
    log.write('LR:%f\n' % args.lr)
    log.write('MIXUP: %d\n' % args.mixup)
    log.write('ALPHA_mixup: %f\n' % args.alpha_mixup)
    log.write('ALPHA_cutmix: %f\n' % args.alpha_cutmix)
    log.write('height: %d\n' % args.height)
    log.write('width: %d\n' % args.width)
    log.write('feather_data_path: %s\n' % args.feather_data_path)
    log.write('image_mode: %s\n'%args.image_mode)
    log.write('schedular: %s\n'%args.LR_SCHEDULER)
    log.write('lr_ratio:%s\n'%args.lr_ratio)
    log.write('patience: %d\n'%args.patience)

    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size

    ## Make sure we are using the GPU . Get CUDA device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    ## Now create the model. Since its greyscale , I have not yet used pretrained model . In Later version ,
    ##I will make the necessary modification to load pretrained weights for greyscale by summing up the weights over one axis or copying greyscale into three channels

    if args.folds == 1:
        in_ch = 1 if args.image_mode == 'gray' else 3
        #in_ch=3
        if args.finetune:
            if args.model.startswith('efficientnet'):
                model = EfficientNet.from_pretrained(args.model, in_channels=in_ch).cuda()
            elif args.model.startswith('se'):
                from NET.seresnet import se50_32_4d_resnext
                model = se50_32_4d_resnext(in_ch=in_ch).cuda()
        #
        else:
            if args.model.startswith('efficientnet'):
                model = EfficientNet.from_name(args.model, in_channels=in_ch).cuda()
            elif args.model.startswith('se'):
                from NET.seresnet import se50_32_4d_resnext
                model = se50_32_4d_resnext(pretrained=None, in_ch=in_ch).cuda()


        model = nn.DataParallel(model)
        #torch.load
        #model.load_state_dict(torch.load('seresnext50_0.5_rotate_liner1_radam_cutmix/global_max_recall.pth'))

        if args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=1e-3)
        elif args.optimizer == 'OVER9000':
            optimizer = Over9000(model.parameters(), lr=args.lr)#, weight_decay=1e-3)  ## New once
        elif args.optimizer == 'RADAM':
            optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)#, weight_decay=1e-3 )

        if args.LR_SCHEDULER == 'COS':
            scheduler = CosineAnnealingLR_with_Restart(optimizer,
                                                       T_max=args.cycle_inter,
                                                       T_mult=1,
                                                       model=model,
                                                       out_dir='../input/',
                                                       take_snapshot=False,
                                                       eta_min=0)
        elif args.LR_SCHEDULER == 'REDUCED':
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-7, patience=args.patience, factor=args.lr_ratio)

        # DATASET SETUP
        csv_path = args.csv_path  # 'BengaliData'
        feather_data_path = args.feather_data_path  # 'BengaliData/feather128'
        train_loader, valid_loader = generate_data_loader(csv_path, feather_data_path, args.batch_size, args.height, args.width,
                                                          num_workers=8, image_mode=args.image_mode)

        ## A very simple loop to train for number of epochs it probably can be made more robust to save only the file with best valid loss
        # history = pd.DataFrame()
        # n_epochs = args.epochs
        valid_recall = 0.0
        best_valid_recall = 0.0
        for num in range(args.cycle_num):
            curr_cycle_best_valid_recall = 0.0
            for epoch in range(args.cycle_inter):
                log.write('\n\nXXXXXXXXXXXXXX--CYCLE NUM: %d  CYCLE INTER:%d --XXXXXXXXXXXXXXXXXXX\n' % (num, epoch))

                # log.write('CURR NUM:%d  CURR INTER:%d\n' % (num, epoch))
                log.write('curr lr: %f\n' % optimizer.param_groups[0]['lr'])
                torch.cuda.empty_cache()
                gc.collect()
                train_recall = train(epoch, train_loader, model, optimizer, criterion, log, args)
                valid_recall = evaluate(epoch, model, criterion, valid_loader, log)

                if valid_recall > curr_cycle_best_valid_recall:
                    log.write(
                        f'Curr validation recall has increased from:  {curr_cycle_best_valid_recall:.4f} to: {valid_recall:.4f}. Saving curr cycle checkpoint\n')
                    torch.save(model.state_dict(),
                               os.path.join(args.outdir,
                                            'current_cycle' + str(
                                                num) + '_max_recall.pth'))  ## Saving model weights based on best validation accuracy.

                    curr_cycle_best_valid_recall = valid_recall
                    # log.write()

                if valid_recall > best_valid_recall:
                    log.write(
                        f'Global validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving global checkpoint\n')
                    torch.save(model.state_dict(),
                               os.path.join(args.outdir,
                                            'global_max_recall.pth'))  ## Saving model weights based on best validation accuracy.
                    best_valid_recall = valid_recall  ## Set the new validation Recall score to compare with next epoch

                if args.LR_SCHEDULER == 'COS':
                    scheduler.step()  ## Want to test with fixed learning rate .If you want to use scheduler please uncomment this .
                elif args.LR_SCHEDULER == 'REDUCED':
                    scheduler.step(valid_recall)

            torch.save(model.state_dict(),
                       os.path.join(args.outdir,
                                    'cycle_' + str(args.cycle_num) + '_last.pth'))

            # history.to_csv(os.path.join(args.outdir, 'log.txt'), index=False)

    else:
        from data import load_feather_data

        nfold = args.folds
        seed = 42
        train_df, data_full = load_feather_data(args.csv_path, args.feather_data_path)
        train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))
        X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] \
                   .values[:, 0], train_df.values[:, 1:]

        train_df['fold'] = np.nan
        # split data
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
        for i, (_, test_index) in enumerate(mskf.split(X, y)):
            train_df.iloc[test_index, -1] = i

        train_df['fold'] = train_df['fold'].astype('int')
        for fold in range(0, nfold):
            log.write('#################FOLD:%d##################\n' % fold)
            val_csv = train_df[train_df['fold'] == fold]
            val_data = data_full[train_df['fold'] == fold]

            train_csv = train_df[train_df['fold'] != fold]
            train_data = data_full[train_df['fold'] != fold]
            
            train_dataset = GraphemeDataset(train_data, train_csv, args.height, args.width, transform=True)
            valid_dataset = GraphemeDataset(val_data, val_csv, args.height, args.width, transform=False)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=False)

            model = EfficientNet.from_pretrained(args.model).cuda()
            model = nn.DataParallel(model)

            if args.optimizer == 'ADAM':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            elif args.optimizer == 'OVER9000':
                optimizer = Over9000(model.parameters(), lr=args.lr, weight_decay=1e-3)  ## New once
            elif args.optimizer == 'RADAM':
                optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            scheduler = CosineAnnealingLR_with_Restart(optimizer,
                                                       T_max=args.cycle_inter,
                                                       T_mult=1,
                                                       model=model,
                                                       out_dir='../input/',
                                                       take_snapshot=False,
                                                       eta_min=0)
            # # DATASET SETUP
            # csv_path = args.csv_path  # 'BengaliData'
            # feather_data_path = args.feather_data_path  # 'BengaliData/feather128'
            # train_loader, valid_loader = generate_data_loader(csv_path, feather_data_path, batch_size=batch_size,
            #                                                   num_workers=8)

            ## A very simple loop to train for number of epochs it probably can be made more robust to save only the file with best valid loss
            # history = pd.DataFrame()
            # n_epochs = args.epochs
            valid_recall = 0.0
            best_valid_recall = 0.0
            for num in range(args.cycle_num):
                curr_cycle_best_valid_recall = 0.0
                for epoch in range(args.cycle_inter):
                    log.write(
                        '\n\nXXXXXXXXXXXXXX--FOLD:%d CYCLE NUM: %d  CYCLE INTER:%d --XXXXXXXXXXXXXXXXXXX\n' % (
                            fold, num, epoch))

                    # log.write('CURR NUM:%d  CURR INTER:%d\n' % (num, epoch))
                    log.write('curr lr: %f\n' % optimizer.param_groups[0]['lr'])
                    torch.cuda.empty_cache()
                    gc.collect()
                    train_recall = train(epoch, train_loader, model, optimizer, criterion, log, args)
                    valid_recall = evaluate(epoch, model, criterion, valid_loader, log)

                    if valid_recall > curr_cycle_best_valid_recall:
                        log.write(
                            f'Fold:{fold:d} curr validation recall has increased from:  {curr_cycle_best_valid_recall:.4f} to: {valid_recall:.4f}. Saving curr cycle checkpoint\n')
                        torch.save(model.state_dict(),
                                   os.path.join(args.outdir,
                                                'Fold_' + str(fold) + '_current_cycle' + str(
                                                    num) + '_max_recall.pth'))  ## Saving model weights based on best validation accuracy.

                        curr_cycle_best_valid_recall = valid_recall
                        # log.write()

                    if valid_recall > best_valid_recall:
                        log.write(
                            f'Fold: {fold:d} global validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving global checkpoint\n')
                        torch.save(model.state_dict(),
                                   os.path.join(args.outdir,
                                                'Fold_' + str(
                                                    fold) + '_global_max_recall.pth'))  ## Saving model weights based on best validation accuracy.
                        best_valid_recall = valid_recall  ## Set the new validation Recall score to compare with next epoch

                    scheduler.step()  ## Want to test with fixed learning rate .If you want to use scheduler please uncomment this .
