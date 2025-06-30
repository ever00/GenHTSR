import os
import numpy as np

from PIL import Image


class DataLoader:
    def __init__(self, dataset_name, img_res=(224, 224), batch_size=128, normalize=True, is_testing=False):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.batch_size = batch_size
        self.normalize = normalize
        self.is_testing = is_testing

        self.train_path = os.path.join('/proj/sciml/users/x_jesst', dataset_name, 'train')  
        self.test_path = os.path.join('/proj/sciml/users/x_jesst', dataset_name, 'test')

        self.train_images = self.load_data(is_testing=False)
        self.test_images = self.load_data(is_testing=True)

        self.n_batches = len(self.train_images) // batch_size if len(self.train_images) > 0 else 0

    def normalize_image(self, img):
        """
        Normalize the image pixel values to [-1, 1]
        """
        return (img / 127.5) - 1 if self.normalize else img

    def load_data(self, is_testing):
        """
        Load all images into memory once
        """
        print('Loading Dataset')
        path = self.test_path if is_testing else self.train_path
        combined_path = os.path.join(path, 'combined')
        undertext_path = os.path.join(path, 'undertext')
        data = []

        for idx, filename in enumerate(os.listdir(combined_path)):

            combined_img = Image.open(os.path.join(combined_path, filename)).convert("RGB").resize((224, 224))
            undertext_img = Image.open(os.path.join(undertext_path, filename.replace('_c_', '_a_'))).convert("RGB").resize((224, 224))

            combined_img = self.normalize_image(np.array(combined_img))
            undertext_img = self.normalize_image(np.array(undertext_img))

            data.append((combined_img, undertext_img, filename)) 

        print('Dataset Loaded')
        return data 

    def load_batch(self, is_testing=False):
        """
        Yield batches of images
        """
        data = self.test_images if is_testing else self.train_images

        if not is_testing:
            np.random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]

            imgs_A, imgs_B, indice = zip(*batch_data)
            imgs_A, imgs_B = np.array(imgs_A), np.array(imgs_B)

            yield imgs_A, imgs_B, indice