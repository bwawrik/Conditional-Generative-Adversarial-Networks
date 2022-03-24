## Some things I needed to install to make this work
#pip install pillow
#pip install colorspacious

#----------------------------------------------------------------------------------------------------------
# First load some images and process them into sub-images of 100x100 pixels
# Color channels are preserved and each image is re-shaped into a np.array of 100x300.
# This is reversed later to restore the image
#----------------------------------------------------------------------------------------------------------
import glob
import numpy as np
import PIL
import math
from PIL import Image
from matplotlib import pyplot as plt

image_path = "/my_image_scrape/images/clouds/*"
file_names = glob.glob(image_path)
for zzz in range(len(file_names)):
    image = Image.open(file_names[zzz])
    imform = image.format
    width, hight = image.size
    mode = image.mode
    myArr = []
    subimage_width = 100
    if width >=subimage_width and hight >=subimage_width:
        for x in range (math.floor(width/50)-1):
            for y in range (math.floor(hight/50)-1):
                im1 = image.crop((x*50, y*50, (x*50)+subimage_width, (y*50)+subimage_width )) #.convert("L")
                #im1.show()
                im1np = np.array(im1).reshape(subimage_width,subimage_width*3)
                myArr.append(im1np)
                print (type(im1np), im1np.shape)
myArrNP = np.array(myArr)


#----------------------------------------------------------------------------------------------------------
# This is the code for the GANN. With more horsepower larger images could be processed and generated I'm
# sure. My computer was limited to 100x100 pixels, preserving the color channels.
#----------------------------------------------------------------------------------------------------------

import numpy as np
from numpy.random import rand
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from numpy import hstack
#from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras import optimizers
from keras.utils.vis_utils import plot_model
import graphviz 
import pydot
from keras.datasets.fashion_mnist import load_data

class cGAN:

    def __init__(self, name, image_file):
        self.name = name
        self.image_file = image_file
        self.multiple = 1
        self.training_epochs = 75
        self.input_image_WH = 100
    
    def getTrainingData (self):
        numpics = len(self.image_file)
        nny = ones((numpics, 1))
        trainX = self.image_file
        self.trainy = nny
        X = np.expand_dims(trainX, axis=-1)
        X = X.astype('float32')
        X = (X - 127.5) / 127.5
        self.trainX = X
        print('Train', self.trainX.shape, self.trainy.shape)
           
    def showImageSubset(self, images):
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.axis('off')
            showImage = images[i]
            showImage = (showImage * 127.5).astype('int')
            showImage = showImage +127
            #print(showImage)
            #plt.imshow(images[i], cmap='gray_r')
            plt.imshow(showImage.reshape(100,100,3))
        print ('TRAINING DATA EXAMPLES:')
        plt.show()
        
    def define_discriminator(self, in_shape=(100,300,1)):
        model = Sequential()
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self, latent_dim):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 10 * 30
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((10, 30, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(5,5), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # generate
        model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
        return model
    
    def define_gan(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(generator)
        # add the discriminator
        model.add(discriminator)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def generate_latent_points(self, latent_dim, n_samples):
        x_input = randn(latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        x_input = self.generate_latent_points(latent_dim, n_samples)
        X = generator.predict(x_input)
        y = zeros((n_samples, 1))
        return X, y

    def generate_real_samples(self, dataset, n_samples):
        f = int(self.trainX.shape[0])
        ix =[]
        for xx in range(n_samples):
            ix.append(randint(0, f))
        X = dataset[ix]
        y = ones((n_samples, 1))
        return X, y

    def train_cGAN(self, g_model, d_model, gan_model, latent_dim, dataset, n_batch=100):
        n_epochs=self.training_epochs
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        for i in range(n_epochs):
            for j in range(bat_per_epo):
                X_real, y_real = self.generate_real_samples(dataset, half_batch)
                d_loss1, _ = d_model.train_on_batch(X_real, y_real)
                X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
                X_gan = self.generate_latent_points(latent_dim, n_batch)
                y_gan = ones((n_batch, 1))
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            latPo = self.generate_latent_points(latent_dim, 100)
            predicted_clothes = self.generator.predict(latPo)
            self.showImageSubset(predicted_clothes)
        #g_model.save('generator.h5')

    def run(self):
        
        latent_dim = 100
        self.getTrainingData()
        self.showImageSubset(self.trainX)
        self.discriminator = self.define_discriminator()
        self.generator     = self.define_generator(latent_dim)
        self.GAN           = self.define_gan(self.generator, self.discriminator)
        self.train_cGAN(self.generator, self.discriminator, self.GAN, latent_dim, self.trainX)
        latPo = self.generate_latent_points(latent_dim, 100)
        predicted_clothes = self.generator.predict(latPo)
        self.showImageSubset(predicted_clothes)
        return True
        
mycGAN = cGAN ('my class', myArrNP)
mycGAN.run()
