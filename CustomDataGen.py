
import tensorflow as tf
import numpy as np
import cv2
import os
import imgaug.augmenters as iaa
import random
from sklearn.utils import shuffle


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self,  df, args, mode):

        self.df = df.copy()
        self.args = args
        self.mode = mode
        self.n = len(self.df)
        if mode == "training" and self.args.aug == True:
            self.augmentation = True
        else:
            self.augmentation = False

    def pre_check_images(self):
        rem_ind_lst = []
        if self.args.num_input_objects < 2:
            print("running single input image mode")
        elif self.args.num_input_objects > 1:
            print("running dual input image mode")
        for ind in range(self.df.shape[0]):
            IMG_PATH = os.path.join(
                self.args.dataset_path, self.df[self.args.image_col].iloc[ind])
            if not os.path.exists(IMG_PATH):
                rem_ind_lst.append(ind)
                continue
            if self.args.model_type == 'Segmentation':
                Mask_PATH = os.path.join(
                    self.args.masks_path, self.df[self.args.image_col].iloc[ind])
                if not os.path.exists(Mask_PATH):
                    rem_ind_lst.append(ind)
                    continue
            if self.args.num_input_objects == 2:

                IMG_PATH = os.path.join(
                    self.args.dataset_path, self.df[self.args.second_image_col].iloc[ind])
                if not os.path.exists(IMG_PATH):
                    rem_ind_lst.append(ind)
                    continue

        self.df.drop(self.df.index[rem_ind_lst], inplace=True)
        self.n = len(self.df)  # append length of dataset list

    def on_epoch_end(self):
        print('end of epoch')
        if self.args.shuffle:
            self.df = shuffle(self.df)

    def __get_input_image(self, path):
        if self.args.image_format == "image":

            if self.args.img_colour:
                image = cv2.imread(path)
                org_height, org_width, nc = image.shape
            else:
                image = cv2.imread(path, 0)
                image = np.expand_dims(image, axis=-1)
                org_height, org_width, nc = image.shape

        elif self.args.image_format == "numpy":
            if self.args.img_colour:
                image = np.load(path)
                image = image/np.max(image)
                image = image*255
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                org_height, org_width, nc = image.shape
            else:
                image = np.load(path)
                image = np.expand_dims(image, axis=-1)
                org_height, org_width, nc = image.shape

        image_r = cv2.resize(image, (self.args.img_size, self.args.img_size),
                             interpolation=cv2.INTER_CUBIC)  # set to 256x256

        image_r = image_r/np.max(image_r)

        return image_r, org_height, org_width

    def Class_Count(self):
        C_count = np.zeros(len(self.args.prediction_classes))
        if self.args.model_type == 'Classification':
            
            for ind in range(len(self.args.prediction_classes)):
                C_count[ind] = self.df[self.df[self.args.prediction_classes[ind]] == 1].shape[0]
        return C_count

    def __get_input_mask(self, path, mask_label):

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.args.img_size, self.args.img_size),
                          interpolation=cv2.INTER_CUBIC)  # set to 256x256
        mask[mask >= 1] = mask_label
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)

        if self.args.image_format == "image":

            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.args.image_format == "numpy":

            mask = np.load(path)

        mask = cv2.resize(mask, (self.args.img_size, self.args.img_size),
                          interpolation=cv2.INTER_CUBIC)  # set to 256x256
        mask[mask >= 1] = mask_label
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)
        return mask

    def get_image(self, path):

        if self.args.image_format == "image":

            if self.args.img_colour:
                image = cv2.imread(path)
                org_height, org_width, nc = image.shape
            else:
                image = cv2.imread(path, 0)
                image = np.expand_dims(image, axis=-1)
                org_height, org_width, nc = image.shape
        elif self.args.image_format == "numpy":

            image = np.load(path)
            image = np.expand_dims(image, axis=-1)
            org_height, org_width, nc = image.shape

        image_r = cv2.resize(image, (self.args.img_size, self.args.img_size),
                             interpolation=cv2.INTER_CUBIC)  # set to 256x256

        image_r = image_r/np.max(image_r)

        return image_r, org_height, org_width

    # Augmentation Horizontal Flip
    def __aug_flip_hr(self, img):
        hflip = iaa.Fliplr(p=1.0)
        img_hf = hflip.augment_image(img)

        return img_hf

    # Augmentation Vertical Flip
    def __aug_flip_vr(self, img):
        vflip = iaa.Flipud(p=1.0)
        img_vf = vflip.augment_image(img)

        return img_vf

    # Augmentation Rotation
    def __aug_rotation(self, img, rot_deg):
        rot1 = iaa.Affine(rotate=(-rot_deg, rot_deg))
        img_rot = rot1.augment_image(img)

        return img_rot

    # Augmentation Cropping
    def __aug_crop(self, img, crop_ratio):
        crop1 = iaa.Crop(percent=(0, crop_ratio))
        img_crop = crop1.augment_image(img)

        return img_crop

    # Augmentation Adding noise
    def __aug_add_noise(self, img, mean_noise, var_noise):
        noise = iaa.AdditiveGaussianNoise(mean_noise, var_noise)
        img = img * 255
        img = img.astype(np.uint8)
        img_noise = noise.augment_image(img)
        img_noise = img_noise / 255
        return img_noise

    # Augmentation Shear
    def __aug_shear(self, img, shear_deg):
        shearX = iaa.ShearX((-shear_deg, shear_deg))
        img_shear = shearX.augment_image(img)

        shearY = iaa.ShearY((-shear_deg, shear_deg))
        img_shear = shearY.augment_image(img_shear)

        return img_shear

    # Augmentation Translation
    def __aug_translation(self, img, trans_pix):
        TransX = iaa.TranslateX(px=(-trans_pix, trans_pix))
        img_trans = TransX.augment_image(img)

        TransY = iaa.TranslateY(px=(-trans_pix, trans_pix))
        img_trans = TransY.augment_image(img_trans)

        return img_trans

    # Augmentation Scale
    def __aug_scale(self, img, scale_ratio):
        ScaleX = iaa.ScaleX((scale_ratio, 3 * scale_ratio))
        img_scale = ScaleX.augment_image(img)
        ScaleY = iaa.ScaleY((scale_ratio, 3 * scale_ratio))
        img_scale = ScaleY.augment_image(img_scale)

        return img_scale

    def batch_augmentation(self, batch_img, random_number):
        for ind in range(batch_img.shape[0]):
            if self.args.img_colour == True:
                img = batch_img[ind, :, :, :]
            else:
                img = batch_img[ind, :, :]
            if random_number == 1:
                img = self.__aug_flip_hr(img)
            elif random_number == 2:
                img = self.__aug_flip_vr(img)
            elif random_number == 3:
                img = self.__aug_rotation(img, 30)
            elif random_number == 4:
                img = self.__aug_crop(img, 0.3)
            elif random_number == 5:
                img = self.__aug_add_noise(img, 100, 50)
            elif random_number == 6:
                img = self.__aug_shear(img, 15)
            elif random_number == 7:
                img = self.__aug_translation(img, 50)
            elif random_number == 8:
                img = self.__aug_scale(img, 0.2)
            if self.args.img_colour == True:
                batch_img[ind, :, :, :] = img
            else:
                batch_img[ind, :, :] = img

            batch_img[ind, :, :] = img

        return batch_img

    def __getitem__(self, index):

        batches = self.df[index *
                          self.args.batch_size:(index + 1) * self.args.batch_size]
        images = []
        images2 = []
        labels = []
        for ind in range(batches.shape[0]):
            if self.args.num_input_objects < 2:
                IMG_PATH = os.path.join(
                    self.args.dataset_path, batches[self.args.image_col].iloc[ind])
                _img, org_height, org_width = self.__get_input_image(IMG_PATH)
                images.append(_img)
                if self.classtype == 'Segmentation':
                    Mask_PATH = os.path.join(
                        self.args.dataset_path, batches[self.args.image_col].iloc[ind])
                    CLASS_LIST = [
                        batches[self.args.prediction_classes].iloc[ind].values]
                    CLASS_LIST = np.array(CLASS_LIST)
                    mask_label = np.argmax(CLASS_LIST) + 1

                    _label = self.__get_input_mask(Mask_PATH, mask_label)
                    labels.append(_label)
                else:
                    labels.append(
                        [batches[self.args.prediction_classes].iloc[ind].values])
            elif self.args.num_input_objects > 1:
                IMG_PATH1 = os.path.join(
                    self.args.dataset_path, batches[self.args.image_col].iloc[ind])
                IMG_PATH2 = os.path.join(
                    self.args.dataset_path, batches[self.args.second_image_col].iloc[ind])
                _img, org_height, org_width = self.__get_input_image(IMG_PATH1)
                _img2, org_height2, org_width2 = self.__get_input_image(
                    IMG_PATH2)
                images.append(_img)
                images2.append(_img2)
                labels.append(
                    [batches[self.args.prediction_classes].iloc[ind].values])

        X = np.array(images, dtype=np.float32)
        if self.args.num_input_objects > 1:
            Z = np.array(images2, dtype=np.float32)

        if self.args.model_type == 'Classification':
            y = np.array(labels, dtype=np.int64)

            y = y.squeeze(axis=1)

        else:
            y = np.array(labels, dtype=np.float32)
        if self.args.num_input_objects > 1:
            return X, Z, y
        else:
            return X, y

    def __len__(self):
        return self.n // self.args.batch_size
