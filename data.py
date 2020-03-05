## Load Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
from augmix import RandomAugMix



## This library is for augmentations .
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    GaussNoise,
    Blur,
    GaussianBlur,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
    Normalize,
    Cutout,
    CoarseDropout,
)
import warnings
warnings.filterwarnings('ignore')


## Create Data from Parquet file mixing the methods of @hanjoonzhoe and @Iafoss

## Create Crop Function @Iafoss

HEIGHT = 137
WIDTH = 236
SIZE = 224

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def Resize(df,size=128):
    resized = {}
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(137, 236).astype(np.uint8)
    #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = img
        image = cv2.resize(img, (168, 96))
        #image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


def parquet2feather(parquet_path, feather_path):
    ##Feather data generation for all train_data
    for i in range(4):
        data = pd.read_parquet(os.path.join(parquet_path, f'train_image_data_{i}.parquet'))
        data = Resize(data)
        data.to_feather(os.path.join(feather_path, f'train_data_{i}{i}_l.feather'))
        del data
        gc.collect()


def load_feather_data(csv_path, feather_data_path):
    ## Load Feather Data
    train = pd.read_csv(os.path.join(csv_path, "train.csv"))
    data0 = pd.read_feather(os.path.join(feather_data_path, "train_data_00_l.feather"))
    data1 = pd.read_feather(os.path.join(feather_data_path, 'train_data_11_l.feather'))
    data2 = pd.read_feather(os.path.join(feather_data_path, 'train_data_22_l.feather'))
    data3 = pd.read_feather(os.path.join(feather_data_path, 'train_data_33_l.feather'))
    data_full = pd.concat([data0, data1, data2, data3], ignore_index=True)
    del data0, data1, data2, data3
    gc.collect()
    print('data full shape:', data_full.shape)
    return train, data_full

## Add Augmentations as suited from Albumentations library
class RandomMorph(ImageOnlyTransform):

    def __init__(self, _min=2, _max=4, element_shape=cv2.MORPH_ELLIPSE, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._min = _min
        self._max = _max
        self.element_shape = element_shape

    def apply(self, image, **params):
        arr = np.random.randint(self._min, self._max, 2)
        kernel = cv2.getStructuringElement(self.element_shape, tuple(arr))

        if random.random() > 0.5:
            # make it thinner
            image = cv2.erode(image, kernel, iterations=1)
        else:
            # make it thicker
            image = cv2.dilate(image, kernel, iterations=1)

        return image

def get_transform(image_mode='gray'):
    lst = [
            #RandomMorph(p=0.4),
            ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=1),
            #CoarseDropout(p=0.5),
            #RandomAugMix(severity=3, width=3, alpha=1., p=1.),
            #Cutout(num_holes=16, max_h_size=16, max_w_size=16, p=0.4, fill_value=0),


            # GridDistortion(distort_limit=0.05, border_mode=cv2.BORDER_CONSTANT, value=1, p=1),
            # OpticalDistortion(p=1, distort_limit=0.05, shift_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=1),
            # RandomGamma(p=1),
            # Blur(p=1),
            # GaussianBlur(blur_limit=3, p=1)

            # OneOf([
            #     GridDistortion(distort_limit=0.05, border_mode=cv2.BORDER_CONSTANT, value=1, p=0.5),
            #     OpticalDistortion(p=0.1, distort_limit=0.05, shift_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=1)
            #     ], p=0.5),
            # OneOf([
            #     Blur(p=0.4),
            #     GaussianBlur(blur_limit=3, p=0.4)
            #     ], p=0.3),
            #
            # RandomGamma(p=0.5),
        ]
    # if image_mode == 'rgb':
    #     lst.append(Normalize())
    train_aug = Compose(lst)
    return train_aug


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    #denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)

    img *= std
    img += mean
    return img


class RandomErasing:
    #random_erasing_area_ratio_range

    def __init__(self, p, area_ratio_range=[0.02, 0.4], min_aspect_ratio=0.3, max_attempt=20):
        self.p = p
        self.max_attempt = max_attempt
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break
        return image

## A lot of heavy augmentations

## Create dataset function
class GraphemeDataset(Dataset):
    def __init__(self, df, label, height, width, transform=True,
                 gridmask=False, image_mode='gray',
                 album_transform=None):
        self.df = df
        self.label = label
        self.transform = transform
        self.data = df.iloc[:, 1:].values
        self.random_earse = RandomErasing(p=1)
        self.width = width
        self.height = height
        self.aug = get_transform(image_mode)
        self.image_mode = image_mode
        self.norm = normalize
        self.album_transform = album_transform

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        label1 = self.label.vowel_diacritic.values[idx]
        label2 = self.label.grapheme_root.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        #image = self.df.iloc[idx][1:].values.reshape(128,128).astype(np.float)
        image = self.data[idx, :].reshape(self.height, self.width).astype(np.uint8)
       # image1 = image
       # print('fdsafsafsA:', type(image), image.max(), image.min())
        if self.image_mode == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            if self.aug:
                augment = self.aug(image=image)
                image = augment['image']
        if self.album_transform:
            res = self.album_transform(image=image)
            image = res['image'].astype(np.float32)

        #image = np.clip(image, 0.0, 255.0)
        image = np.asarray(image, dtype=np.float32)
        if self.image_mode == 'gray':
            image /= 255.0
            image = np.expand_dims(image, 0)
        else:
            image = self.norm(image)
            image = np.transpose(image, (2, 0, 1))


        return image, label1, label2, label3#, image1


def generate_train_val_dataset(csv_path, feather_path, height, width, debug=False, image_mode='gray',
                               gridmask=False, gridmask_num_grid=None):
    ## Do a train-valid split of the data to create dataset and dataloader . Specify random seed to get reproducibility

    ###########################################1 sklearn 随机20% data as val set#####################################
    train, data_full = load_feather_data(csv_path, feather_path)
    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(train, test_size=0.20, random_state=42, shuffle=True)  ## Split Labels
    data_train_df, data_valid_df = train_test_split(data_full, test_size=0.20, random_state=42,
                                                    shuffle=True)  ## split data
    del data_full
    gc.collect()

    ############################################1 严格按照子类 随机1/6 data as val set#####################################
    # train_df, data_train_df, valid_df, data_valid_df = split_k_folder(csv_path, feather_path, nfold=6, seed=42)
    if gridmask:
        import albumentations
        num_grid = eval(gridmask_num_grid)
        transform_train = albumentations.Compose([
            GridMask(num_grid=num_grid, p=1),
        ])
    else:
        transform_train = None

    train_dataset = GraphemeDataset(data_train_df, train_df, height, width, transform=True, image_mode=image_mode,
                                    album_transform=transform_train)
    valid_dataset = GraphemeDataset(data_valid_df, valid_df, height, width, transform=False, image_mode=image_mode)
    torch.cuda.empty_cache()
    gc.collect()


    if debug:
        return data_train_df, data_valid_df, train_dataset, valid_dataset
    else:
        return train_dataset, valid_dataset



def split_k_folder(csv_path, feather_data_path, nfold=5, seed=12):
    train_df, data_full = load_feather_data(csv_path, feather_data_path)
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

    val_csv = train_df[train_df['fold'] == 0]
    val_data = data_full[train_df['fold'] == 0]

    train_csv = train_df[train_df['fold'] != 0]
    train_data = data_full[train_df['fold'] != 0]

    return train_csv, train_data, val_csv, val_data



##Visulization function for checking Original and augmented image
def visualize(original_image, aug_image, index = 0):
    fontsize = 18

    f, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original image', fontsize=fontsize)
    ax[1].imshow(aug_image, cmap='gray')
    ax[1].set_title('Augmented image', fontsize=fontsize)
    plt.savefig('res'+str(index)+'.jpg')



def generate_data_loader(csv_path, feather_path, batch_size, height, width, num_workers=1, image_mode='gray',
                         gridmask=False, gridmask_num_grid=None):
    ## Create data loader and get ready for training .
    #batch_size = 32

    train_dataset, valid_dataset = generate_train_val_dataset(csv_path, feather_path, height, width, image_mode=image_mode,
                                                              gridmask=False, gridmask_num_grid=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader




def check_aug_data(csv_path, feather_path):
    height=224
    width=224
    train, data_full = load_feather_data(csv_path, feather_path)

    data_train_df, data_valid_df, train_dataset, valid_dataset = generate_train_val_dataset(train, data_full, height, width, debug=True)
    ## One image taken from raw dataframe another from dataset
    orig_image = data_train_df.iloc[0, 1:].values.reshape(height, width).astype(np.float)
    #aug_image = train_dataset[0][0]

    for i in range(20):
        aug_image = train_dataset[0][0]
        #print('shape:', aug_image.shape)

        visualize(orig_image, aug_image, i)
    #visualize(orig_image, aug_image)


def convert2feather():
    parquet_path = '/data1/wangwenpeng/Bengali/BengaliData'
    feather_path = '/data1/wangwenpeng/Bengali/BengaliData/feather224'
    if not os.path.exists(feather_path):
        os.makedirs(feather_path)

    parquet2feather(parquet_path, feather_path)


if __name__ == '__main__':
    csv_path = 'BengaliData'
    feather_path = 'BengaliData/feather_resize128'
    # if not os.path.exists(feather_path):
    #     os.makedirs(feather_path)
    #
    # parquet2feather(csv_path, feather_path)

    # # # split_k_folder(csv_path, feather_path)
    # # # #
    # # # check_aug_data(csv_path, feather_path)
    # # #
    # #

    mode = 'mixup'
    mode = 'cutmix'
    mode = 'FMIX'
    image_mode = 'gray'
    alpha = 1
    from mixup import cutmix, mixup
    tr_loader, val_loader = generate_data_loader(csv_path, feather_path, 10, 128, 128, num_workers=1, image_mode=image_mode)
    for image, label1, label2, label3, image1 in tr_loader:
        if mode=='cutmix':
            image, labels1, labels2, labels3 = cutmix(image, label1, label2, label3, alpha)
        elif mode=='mixup':
            image, labels1, labels2, labels3 = mixup(image, label1, label2, label3, alpha)
        elif mode == 'FMIX':
            from FMix.implementations.lightning import FMix
            fmixer = FMix(alpha=0.5, size=(128,128))
            image = fmixer(image)
        print(image.shape, label1.shape, label2.shape, label3.shape)
        image = image[0].numpy()
        image = np.transpose(image, (1,2,0))

        image1 = image1[0].numpy()
        #image = image*255.0


        #rgb
        if image_mode == 'rgb':
            image = denormalize(image)
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

        else:
            image *= 255.0
            image1 = np.expand_dims(image1, -1)

        cv2.imwrite('test.jpg', np.hstack((image1, image)))


        #print(image[0][0][100][100:150])
        a = input()



