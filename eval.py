import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from data import load_feather_data, GraphemeDataset
from NET.efficientnet import EfficientNet
from metric import macro_recall_multi
import numpy as np


def evaluate(model, criterion, valid_loader):
    # for i in range(len(models)):
    #     models[i].eval()
    losses = []
    accs = []
    recalls = []
    acc = 0.0
    total = 0.0
    # print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0
    with torch.no_grad():
        for idx, (inputs, labels1, labels2, labels3) in enumerate(tqdm(valid_loader)):
            inputs = inputs.cuda()  # to(device)
            labels1 = labels1.cuda()  # to(device)
            labels2 = labels2.cuda()  # to(device)
            labels3 = labels3.cuda()  # to(device)
            total += len(inputs)
            outputs1, outputs2, outputs3 = model(inputs.unsqueeze(1).float())
            loss1 = criterion(outputs1, labels1)
            loss2 = 2 * criterion(outputs2, labels2)
            loss3 = criterion(outputs3, labels3)
            running_loss += loss1.item() + loss2.item() + loss3.item()
            running_recall += macro_recall_multi(outputs2, labels2, outputs1, labels1, outputs3, labels3)
            running_acc += (outputs1.argmax(1) == labels1).float().mean()
            running_acc += (outputs2.argmax(1) == labels2).float().mean()
            running_acc += (outputs3.argmax(1) == labels3).float().mean()
            acc = running_acc / total
            # scheduler.step()
    losses.append(running_loss / len(valid_loader))
    accs.append(running_acc / (len(valid_loader) * 3))
    recalls.append(running_recall / len(valid_loader))
    total_recall = running_recall / len(valid_loader)  ## No its not Arnold Schwarzenegger movie
    print('val acc : {:.2f}%\n'.format(running_acc / (len(valid_loader) * 3)))
    print('loss : {:.4f}\n'.format(running_loss / len(valid_loader)))
    print('recall: {:.4f}\n'.format(running_recall / len(valid_loader)))
    return total_recall







if __name__ == '__main__':
    gpu_ids = '2,5'
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    batch_size = 2048
    csv_path = 'BengaliData'
    feather_path = 'BengaliData/feather128'
    criterion = nn.CrossEntropyLoss()
    #train, data_full = load_feather_data(csv_path, feather_path)
    #dataset = GraphemeDataset(data_full, train, transform=False)
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    model_name = 'efficientnet-b0'
    ckpts = ['eff0_cv5_cos_norm_adam/Fold_0_global_max_recall.pth',
             'eff0_cv5_cos_norm_adam/Fold_1_global_max_recall.pth',
             'eff0_cv5_cos_norm_adam/Fold_2_global_max_recall.pth',
             'eff0_cv5_cos_norm_adam/Fold_3_global_max_recall.pth',
             'eff0_cv5_cos_norm_adam/Fold_4_global_max_recall.pth',
             ]
    models = []

    nfold = 5
    seed = 42
    train_df, data_full = load_feather_data(csv_path, feather_path)
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

    recall_final = 0

    for fold in range(nfold):
        print('#################FOLD:%d##################\n' % fold)
        val_csv = train_df[train_df['fold'] == fold]
        val_data = data_full[train_df['fold'] == fold]

        valid_dataset = GraphemeDataset(val_data, val_csv, transform=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   num_workers=8,
                                                   shuffle=False)
        ckpt = ckpts[fold]
        model = EfficientNet.from_pretrained(model_name).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        recall_final += evaluate(model, criterion, valid_loader)

    print('recall_final:', recall_final/nfold)



















