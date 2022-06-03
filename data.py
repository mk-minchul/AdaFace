import os
import torchvision.datasets as datasets

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
import evaluate_utils

class DataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.data_root = kwargs['data_root']
        self.train_data_path = kwargs['train_data_path']
        self.val_data_path = kwargs['val_data_path']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.train_data_subset = kwargs['train_data_subset']

        self.low_res_augmentation_prob = kwargs['low_res_augmentation_prob']
        self.crop_augmentation_prob = kwargs['crop_augmentation_prob']
        self.photometric_augmentation_prob = kwargs['photometric_augmentation_prob']

        concat_mem_file_name = os.path.join(self.data_root, self.val_data_path, 'concat_validation_memfile')
        self.concat_mem_file_name = concat_mem_file_name


    def prepare_data(self):
        # call this once to convert val_data to memfile for saving memory
        if not os.path.isdir(os.path.join(self.data_root, self.val_data_path, 'agedb_30', 'memfile')):
            print('making validation data memfile')
            evaluate_utils.get_val_data(os.path.join(self.data_root, self.val_data_path))

        if not os.path.isfile(self.concat_mem_file_name):
            # create a concat memfile
            concat = []
            for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
                np_array, issame = evaluate_utils.get_val_pair(path=os.path.join(self.data_root, self.val_data_path),
                                                               name=key,
                                                               use_memfile=False)
                concat.append(np_array)
            concat = np.concatenate(concat)
            evaluate_utils.make_memmap(self.concat_mem_file_name, concat)


    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            print('creating train dataset')
            self.train_dataset = train_dataset(self.data_root,
                                               self.train_data_path,
                                               self.low_res_augmentation_prob,
                                               self.crop_augmentation_prob,
                                               self.photometric_augmentation_prob,
                                               )

            # checking same list for subseting
            if self.train_data_path == 'faces_emore/imgs' and self.train_data_subset:
                with open('ms1mv2_train_subset_index.txt', 'r') as f:
                    subset_index = [int(i) for i in f.read().split(',')]

                # remove too few example identites
                self.train_dataset.samples = [self.train_dataset.samples[idx] for idx in subset_index]
                self.train_dataset.targets = [self.train_dataset.targets[idx] for idx in subset_index]
                value_counts = pd.Series(self.train_dataset.targets).value_counts()
                to_erase_label = value_counts[value_counts<5].index
                e_idx = [i in to_erase_label for i in self.train_dataset.targets]
                self.train_dataset.samples = [i for i, erase in zip(self.train_dataset.samples, e_idx) if not erase]
                self.train_dataset.targets = [i for i, erase in zip(self.train_dataset.targets, e_idx) if not erase]

                # label adjust
                max_label = np.max(self.train_dataset.targets)
                adjuster = {}
                new = 0
                for orig in range(max_label+1):
                    if orig in to_erase_label:
                        continue
                    adjuster[orig] = new
                    new += 1

                # readjust class_to_idx
                self.train_dataset.targets = [adjuster[orig] for orig in self.train_dataset.targets]
                self.train_dataset.samples = [(sample[0], adjuster[sample[1]]) for sample in self.train_dataset.samples]
                new_class_to_idx = {}
                for label_str, label_int in self.train_dataset.class_to_idx.items():
                    if label_int in to_erase_label:
                        continue
                    else:
                        new_class_to_idx[label_str] = adjuster[label_int]
                self.train_dataset.class_to_idx = new_class_to_idx

            print('creating val dataset')
            self.val_dataset = val_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


def train_dataset(data_root, train_data_path,
                  low_res_augmentation_prob,
                  crop_augmentation_prob,
                  photometric_augmentation_prob):

    train_dir = os.path.join(data_root, train_data_path)
    train_dataset = CustomImageFolderDataset(root=train_dir,
                                             transform=transforms.Compose([
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                             ]),
                                             low_res_augmentation_prob=low_res_augmentation_prob,
                                             crop_augmentation_prob=crop_augmentation_prob,
                                             photometric_augmentation_prob=photometric_augmentation_prob,
                                             )

    return train_dataset


def val_dataset(data_root, val_data_path, concat_mem_file_name):
    val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
    # theses datasets are already normalized with mean 0.5, std 0.5
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
    val_data_dict = {
        'agedb_30': (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)

    return val_dataset


def test_dataset(data_root, val_data_path, concat_mem_file_name):
    val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
    # theses datasets are already normalized with mean 0.5, std 0.5
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
    val_data_dict = {
        'agedb_30': (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
    return val_dataset


class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 ):
        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.low_res_augmentation_prob = low_res_augmentation_prob
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

        self.tot_rot_try = 0
        self.rot_success = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if 'WebFace' in self.root:
            # swap rgb to bgr since image is in rgb for webface
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample, _ = self.augment(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def augment(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(sample))
            if hasattr(F, '_get_image_size'):
                orig_W, orig_H = F._get_image_size(sample)
            else:
                # torchvision 0.11.0 and above
                orig_W, orig_H = F.get_image_size(sample)
            i, j, h, w = self.random_resized_crop.get_params(sample,
                                                            self.random_resized_crop.scale,
                                                            self.random_resized_crop.ratio)
            cropped = F.crop(sample, i, j, h, w)
            new[i:i+h,j:j+w, :] = np.array(cropped)
            sample = Image.fromarray(new.astype(np.uint8))
            crop_ratio = min(h, w) / max(orig_H, orig_W)
        else:
            crop_ratio = 1.0

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))
        else:
            resize_ratio = 1

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                                  self.photometric.saturation, self.photometric.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    sample = F.adjust_brightness(sample, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    sample = F.adjust_contrast(sample, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    sample = F.adjust_saturation(sample, saturation_factor)

        information_score = resize_ratio * crop_ratio
        return sample, information_score


class FiveValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):
        '''
        concatenates all validation datasets from emore
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
        }
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        '''
        self.dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}

        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
        assert key_orders == ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = evaluate_utils.read_memmap(concat_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame)
        assert len(self.all_issame) == len(self.all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)


def low_res_augmentation(img):
    # resize the image to a small size and enlarge it back
    img_shape = img.shape
    side_ratio = np.random.uniform(0.2, 1.0)
    small_side = int(side_ratio * img_shape[0])
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

    return aug_img, side_ratio