import os
import numpy as np

from PIL import Image


class DataLoader:
    def __init__(self, dataset_name, img_res=(28, 28), batch_size=128, normalize=True, is_testing=False):
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
            combined_img = Image.open(os.path.join(combined_path, filename))
            undertext_img = Image.open(os.path.join(undertext_path, filename)).convert("RGB")

            combined_img = self.normalize_image(np.array(combined_img))
            undertext_img = self.normalize_image(np.array(undertext_img))

            label = filename[-5:-4] 
            indice = "_".join(filename.split("_")[:-1])

            data.append((combined_img, undertext_img, label, indice)) 
        print(f"Dataset Loaded: {len(data)} images")
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

            combined_imgs, undertext_imgs, labels, indices = zip(*batch_data) 
            combined_imgs, undertext_imgs = np.array(combined_imgs), np.array(undertext_imgs)
            yield combined_imgs, undertext_imgs, labels, indices
