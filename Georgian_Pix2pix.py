import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import datetime
import os
import gc

from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

import sys
sys.path.append("/proj/sciml/users/x_jesst")
from Georgian_data_loader import DataLoader


class Pix2Pix():
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset_name = 'synthetic_Georgian'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        self.gf = 64
        self.df = 64

        optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        combined_img = Input(shape=self.img_shape) 
        undertext_img = Input(shape=self.img_shape) 

        fake_A = self.generator(combined_img)

        self.discriminator.trainable = False
        valid = self.discriminator([combined_img, fake_A])

        self.combined = Model(inputs=[combined_img, undertext_img], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 10], optimizer=optimizer)


    def build_generator(self):
        """
        U-Net Generator
        """

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """
            Layers used during downsampling
            """
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """
            Layers used during upsampling
            """
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input]) 
    
            return u

        d0 = Input(shape=self.img_shape)

        d1 = conv2d(d0, self.gf, bn=False)      
        d2 = conv2d(d1, self.gf * 2)            
        d3 = conv2d(d2, self.gf * 4)            
        d4 = conv2d(d3, self.gf * 8) 
        d5 = conv2d(d4, self.gf * 8)

      
        u4 = deconv2d(d5, d4, self.gf * 8)
        u3 = deconv2d(u4, d3, self.gf * 8)      
        u2 = deconv2d(u3, d2, self.gf * 4)      
        u1 = deconv2d(u2, d1, self.gf * 2)      
        u0 = deconv2d(u1, d0, self.gf)         

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u0)
        return Model(d0, output_img)


    def build_discriminator(self):
        """
        PatchGAN Discriminator
        """

        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)  
        img_B = Input(shape=self.img_shape)  
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=128, sample_interval=128):
        '''
        train loop
        '''
        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)  # Real
        fake = np.zeros((batch_size,) + self.disc_patch)  # Fake

        for epoch in range(epochs):
            self.sample_images(epoch)

            for batch_i, (combined_img, undertext_img, _) in enumerate(self.data_loader.load_batch()):
                if combined_img.size == 0:  
                    continue

                batch_size_actual = combined_img.shape[0]
                valid = np.ones((batch_size_actual,) + self.disc_patch, dtype=np.float32)
                fake = np.zeros((batch_size_actual,) + self.disc_patch, dtype=np.float32)

                fake_A = self.generator.predict(combined_img)

                d_loss_real = self.discriminator.train_on_batch([combined_img, undertext_img], valid[:batch_size_actual])
                d_loss_fake = self.discriminator.train_on_batch([combined_img, fake_A], fake[:batch_size_actual])
                d_loss = np.add(d_loss_real, d_loss_fake) / 2

                g_loss = self.combined.train_on_batch([combined_img, undertext_img], [valid[:batch_size_actual], undertext_img])

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss[0], 100 * d_loss[1], g_loss[0], elapsed_time))

                del combined_img, undertext_img, fake_A, valid, fake
                gc.collect()

            if epoch % 5 == 0:
                tf.keras.backend.clear_session()
                gc.collect()


    def sample_images(self, epoch):
        '''
        Sample images during training to plot generation progression
        '''
        os.makedirs('pix2pix_Georgian', exist_ok=True)
        r, c = 3, 3
        batch_data = next(self.data_loader.load_batch())

        combined_img, undertext_img, _ = batch_data
        fake_combined = self.generator.predict(combined_img)
        concat_imgs = np.concatenate([undertext_img[:3], fake_combined[:3], combined_img[:3]])
        concat_imgs = 0.5 * concat_imgs + 0.5
        titles = ['Ground Truth'] * 3 + ['Generated'] * 3 + ['Input'] * 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if i == 1:
                    gray_img = Image.fromarray((concat_imgs[cnt] * 255).astype(np.uint8))
                    axs[i,j].imshow(gray_img)
                else:
                    axs[i,j].imshow(concat_imgs[cnt])
                axs[i,j].set_title(titles[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"pix2pix_Georgian/{epoch}.png")
        plt.close()


    def generate_and_save_test_set(self, output_dir):
        '''
        Generate clean under-text images from test set
        '''
        os.makedirs(output_dir, exist_ok=True)
        for idx, (combined_img, _, indices) in enumerate(self.data_loader.load_batch(is_testing=True)):

            fake_A = self.generator.predict(combined_img)
            fake_A = 0.5 * fake_A + 0.5  # Rescale to [0, 1]
            fake_A = (fake_A * 255).astype(np.uint8)

            for i in range(len(combined_img)):
                img_name = os.path.join(output_dir, f"{indices[i]}.png")
                fake_A_gray = Image.fromarray(fake_A[i])
                fake_A_gray.save(img_name)
            print(f"Processed batch {idx + 1}")


if __name__ == '__main__':
    gan = Pix2Pix()

    gan.train(epochs=30, batch_size=128, sample_interval=128)

    dirpath = '/proj/sciml/users/x_jesst/pix2pix/pix2pix_Georgian'
    gan.generate_and_save_test_set(dirpath)
