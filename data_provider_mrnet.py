import os
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from torch.utils.data import DataLoader


def get_image(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose((1, 2, 0)).astype(np.float32)
    img = np.clip(img, 0, 1200)
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

def load_data(axial_path, coronal_path, sagittal_path, label, patient, subset):
    axial_img = get_image(axial_path)
    coronal_img = get_image(coronal_path)
    sagittal_img = get_image(sagittal_path)

    img_out = np.concatenate((axial_img, coronal_img, sagittal_img), axis=0)
    if subset == 'train':
        img_out = process(img_out)
    else:
        img_out = multimodal_Standard(img_out)

    label = np.array(label)

    return img_out, label, patient


def calc_label(grade_path_name, labels, num_0, num_1):
    if grade_path_name == '0':
        labels.append(0)
        num_0 += 1

    if grade_path_name == '1':
        labels.append(1)
        num_1 += 1

    return labels, num_0, num_1


class TB_Dataset(Dataset):

    def __init__(self, subset, data_path, if_offline_data_aug, ifbalanceloader, test_type, image_type, num_classes, ifonline_aug, val=False, online_aug_type=None,args=False):
        super(TB_Dataset, self).__init__()
        self.subset = subset
        self.test_type = test_type
        self.image_type = image_type
        self.num_classes = num_classes
        self.ifonline_aug = ifonline_aug
        self.online_aug_type = online_aug_type
        self.args = args

        # dataset path
        if not val:
            if 'train' in subset:
                source_path = os.path.join(data_path, 'train')
            elif 'test' in subset:
                source_path = os.path.join(data_path, 'test')
            print(source_path)
        else:
            source_path = os.path.join(data_path, 'test')

        patients = []
        labels = []
        axial = []
        coronal = []
        sagittal = []

        num_0 = 0
        num_1 = 0

        for grade_path_name in os.listdir(source_path):
            grade_path = os.path.join(source_path, grade_path_name)
            for patient_path_name in os.listdir(grade_path):
                patient_path = os.path.join(grade_path, patient_path_name)
                if if_offline_data_aug:
                    for nii_path_name in os.listdir(patient_path):
                        if nii_path_name.startswith('axial'):
                            axial_path = os.path.join(patient_path, nii_path_name)
                            axial.append(axial_path)
                            patients.append(patient_path_name)
                            coronal_path = os.path.join(patient_path, 'coronal' + str(nii_path_name.split('axial')[1]))
                            if os.path.exists(coronal_path):
                                coronal.append(coronal_path)
                            else:
                                print('not exist ' + coronal_path)
                            sagittal_path = os.path.join(patient_path, 'sagittal' + str(nii_path_name.split('axial')[1]))
                            if os.path.exists(sagittal_path):
                                sagittal.append(sagittal_path)
                            else:
                                print('not exist ' + sagittal_path)
                            labels, num_0, num_1 = calc_label(grade_path_name, labels, num_0, num_1)

                else:
                    axial_path = os.path.join(patient_path, 'axial.nii.gz')
                    if os.path.exists(axial_path):
                        axial.append(axial_path)
                        patients.append(patient_path_name)

                        # coronal
                        coronal_path = os.path.join(patient_path, 'coronal' + str(axial_path.split('axial')[1]))
                        if os.path.exists(coronal_path):
                            coronal.append(coronal_path)
                        else:
                            print('not exist ' + coronal_path)

                        # sagittal
                        sagittal_path = os.path.join(patient_path, 'sagittal' + str(axial_path.split('axial')[1]))
                        if os.path.exists(sagittal_path):
                            sagittal.append(sagittal_path)
                        else:
                            print('not exist ' + sagittal_path)

                        labels, num_0, num_1 = calc_label(grade_path_name, labels, num_0, num_1)
                    else:
                        print('not exist ' + axial_path)

        if not ifbalanceloader:
            if not val:
                print('Num of all samples:', len(labels))
                print('Num of label 0:', num_0)
                print('Num of label 1:', num_1)


        self.axial = axial
        self.coronal = coronal
        self.sagittal = sagittal
        self.labels = labels
        self.patients = patients

    def __getitem__(self, index):

        img_out, label, patient = load_data(self.axial[index], self.coronal[index], self.sagittal[index], self.labels[index], self.patients[index], self.subset)
        return img_out, label, patient

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    data_path = '/mnt/hard_disk/liutianling/Invasion/BBox/split/axial+coronal+sagittal_cv3folder'
    fold = '3fold'
    use_path = os.path.join(data_path, fold)
    batch_size = 32
    train_set = TB_Dataset('train', use_path, True, False, False, 'bbox', num_classes=2, ifonline_aug=False)
    dataloaders = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)
    for image, label, patient in dataloaders:
        print(image.shape)