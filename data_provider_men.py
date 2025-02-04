import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


clip_value = 1200

def get_image(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose((1, 2, 0)).astype(float)
    img = np.clip(img, 0, clip_value)
    img = np.expand_dims(img, axis=0)
    return img

def Standardize(images):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    new: z-score is used but keep the background with zero!
    """
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    mask_location = images.sum(0) > 0
    # print(mask_location.shape)
    for k in range(images.shape[0]):
        image = images[k,...]
        image = np.array(image, dtype='float32')
        mask_area = image[mask_location]
        image[mask_location] -= mask_area.mean()
        image[mask_location] /= mask_area.std()
        images[k,...] = image
    # return (image - self.mean) / np.clip(self.std, a_min=self.eps, a_max=None)
    # print(images.mean(),images.std())
    # print(images.shape)
    return images

def multimodal_Standard(image):
    for i in range(image.shape[0]):
        img = image[i, ...]
        new_img = Standardize(img)
        if i == 0:
            out_image = new_img
        else:
            out_image = np.concatenate((out_image, new_img), axis=0)
    out_image = out_image.copy()
    return out_image

def process(out_image):
    if out_image.shape[0] == 1:
        new_out_img = Standardize(out_image)
        new_out_img = new_out_img.copy()
    else:
        for i in range(out_image.shape[0]):
            new_img = out_image[i, ...]
            new_img = Standardize(new_img)
            if i == 0:
                new_out_img = new_img
            else:
                new_out_img = np.concatenate((new_out_img, new_img), axis=0)
    return new_out_img

def load_data(t1_path, t2_path, adc_path, label, patient, subset):
    t1_img = get_image(t1_path)
    t2_img = get_image(t2_path)
    adc_img = get_image(adc_path)

    img_out = np.concatenate((t1_img, t2_img, adc_img), axis=0)
    if subset == 'train':
        img_out = process(img_out)
    else:
        img_out = multimodal_Standard(img_out)

    label = np.array(label)

    return img_out, label, patient


def calc_label(grade_path_name, labels, num_0, num_1, num_2):
    if grade_path_name == 'Grade_1':
        labels.append(0)
        num_0 += 1

    if grade_path_name == 'Grade_2_invasion':
        labels.append(1)
        num_1 += 1

    if grade_path_name == 'Grade_2_noninvasion':
        labels.append(2)
        num_2 += 1

    return labels, num_0, num_1, num_2


class TB_Dataset(Dataset):

    def __init__(self, subset, data_path, if_offline_data_aug, ifbalanceloader, test_type, image_type, num_classes, ifonline_aug, val=False, args=False):
        super(TB_Dataset, self).__init__()
        self.subset = subset
        self.test_type = test_type
        self.image_type = image_type
        self.num_classes = num_classes
        self.ifonline_aug = ifonline_aug
        self.args = args

        # dataset path
        if val:
            source_path = os.path.join(data_path, 'test')
        else:
            if 'train' in subset:
                source_path = os.path.join(data_path, 'train')
            else:
                source_path = os.path.join(data_path, 'test')
            print(source_path)

        patients = []
        labels = []
        t1 = []
        t2 = []
        adc = []

        num_0 = 0
        num_1 = 0
        num_2 = 0

        for grade_path_name in os.listdir(source_path):
            grade_path = os.path.join(source_path, grade_path_name)
            for patient_path_name in os.listdir(grade_path):
                patient_path = os.path.join(grade_path, patient_path_name)
                if if_offline_data_aug:
                    for nii_path_name in os.listdir(patient_path):
                        if image_type == 'bbox':
                            if nii_path_name.startswith('t1_bbox'):
                                t1_path = os.path.join(patient_path, nii_path_name)
                                t1.append(t1_path)
                                patients.append(patient_path_name)
                                # t2
                                t2_path = os.path.join(patient_path, 't2_bbox' + str(nii_path_name.split('t1_bbox')[1]))
                                if os.path.exists(t2_path):
                                    t2.append(t2_path)
                                else:
                                    print('not exist ' + t2_path)
                                # adc
                                adc_path = os.path.join(patient_path, 'adc_bbox' + str(nii_path_name.split('t1_bbox')[1]))
                                if os.path.exists(adc_path):
                                    adc.append(adc_path)
                                else:
                                    print('not exist ' + adc_path)

                                labels, num_0, num_1, num_2 = calc_label(grade_path_name, labels, num_0, num_1, num_2)

                        else:
                            print('wrong type!')
                else:
                    if image_type == 'bbox':
                        t1_path = os.path.join(patient_path, 't1_bbox.nii.gz')
                        if os.path.exists(t1_path):
                            t1.append(t1_path)
                            patients.append(patient_path_name)

                            # t2
                            t2_path = os.path.join(patient_path, 't2_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(t2_path):
                                t2.append(t2_path)
                            else:
                                print('not exist ' + t2_path)

                            # adc
                            adc_path = os.path.join(patient_path, 'adc_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(adc_path):
                                adc.append(adc_path)
                            else:
                                print('not exist ' + adc_path)

                            labels, num_0, num_1, num_2 = calc_label(grade_path_name, labels, num_0, num_1, num_2)
                        else:
                            print('not exist ' + t1_path)

        if not ifbalanceloader:
            if not val:
                print('Num of all samples:', len(labels))
                print('Num of label 0 (Grade_1):', num_0)
                print('Num of label 1 (Grade_2_invasion):', num_1)
                print('Num of label 2 (Grade_2_noninvasion):', num_2)


        self.t1 = t1
        self.t2 = t2
        self.adc = adc
        self.labels = labels
        self.patients = patients

    def __getitem__(self, index):

        img_out, label, patient = load_data(self.t1[index], self.t2[index], self.adc[index], self.labels[index], self.patients[index], self.subset)
        return img_out, label, patient

    def __len__(self):
        return len(self.labels)
