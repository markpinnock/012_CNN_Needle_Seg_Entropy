import matplotlib.pyplot as plt
import numpy as np
import os
# import scipy.io as scio
import sys
import tensorflow as tf
import tensorflow.keras as keras

from NetworkLayers import UNetGen

sys.path.append('..')


""" Visualise kernels (gradient ascent technique) """
# TODO: NEEDS CLEAN UP - COMBINE WITH OTHER VISUALISATION SCRIPTS

LO_VOL_SIZE = (512, 512, 3, 1, )
NC = 4
LAYER = 4
FILTER = 3
ETA = 1

MODEL_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/012_CNN_Needle_Seg_Entropy/models/Phase_2/"
lambd = "10.0"

UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
print(UNet.summary())
UNet.load_weights(f"{MODEL_PATH}nc4_ep100_eta0.001_lam{lambd}/nc4_ep100_eta0.001_lam{lambd}.ckpt")
# print([var.shape for var in UNet.trainable_variables])
input_image = np.load('test2.npy')
input_image = (input_image + 2917) / (16297 + 2917)
bottom_layer = UNet(input_image[np.newaxis, :, :, :, np.newaxis])[LAYER]
# print([layer.numpy().shape for layer in layers])

entropy_vec = np.zeros(64)
KLD_vec = np.zeros(64)

for i in range(64):
    input_noise = tf.random.uniform([1, 64, 64, 3, 1], -1, 1)
    KLD_reference_distribution = input_noise

    for j in range(5):
        with tf.GradientTape() as tape:
            tape.watch(input_noise)
            output = UNet(input_noise)[LAYER]
            loss = tf.reduce_mean(output[:, :, :, :, i])
            grad = tape.gradient(loss, input_noise)
            grad_norm = grad / (tf.sqrt(tf.reduce_mean(tf.square(grad))) + 1e-5)
            input_noise += grad_norm * ETA
            print(i, j)

    kernel = np.squeeze(input_noise.numpy())

    KLD_counts, _ = np.histogram(KLD_reference_distribution, 64)
    KLD_prob = KLD_counts / np.sum(KLD_counts)

    counts, _ = np.histogram(kernel, 64)
    prob = counts / np.sum(counts)
    prob = prob[np.nonzero(prob)]
    KLD_prob = KLD_prob[np.nonzero(prob)]
    prob_ratio = KLD_prob / prob
    prob_ratio = prob_ratio[np.nonzero(prob_ratio)]

    entropy_vec[i] = -np.sum(prob * np.log2(prob))
    KLD_vec[i] = -np.sum(prob * np.log2(prob_ratio))

    # plt.figure(1)
    # plt.subplot(8, 8, i+1)
    # plt.imshow(kernel[:, :, 1], cmap='hot')
    # plt.gca().set_title(f"{entropy_vec[i]:.2f}")
    # plt.axis('off')

    # plt.figure(2)
    # plt.subplot(8, 8, i+1)
    # plt.hist(np.ravel(kernel))

    plt.figure(3)
    plt.subplot(8, 16, 2*(i+1)-1)
    plt.imshow(np.fliplr(bottom_layer[0, :, :, 1, i].numpy().T), cmap='gray', origin='lower')
    plt.axis('off')
    plt.subplot(8, 16, 2*(i+1))
    plt.imshow(kernel[:, :, 1], cmap='hot')
    plt.axis('off')

plt.figure(4)
plt.hist(entropy_vec)
plt.plot(entropy_vec, np.ones(entropy_vec.shape), 'rx')
plt.plot(np.ones(25) * np.median(entropy_vec), np.linspace(0, 25, 25), 'k-')
plt.plot(np.ones(25) * np.mean(entropy_vec), np.linspace(0, 25, 25), 'g-')
plt.xlim([0, 7])
plt.title('Entropy')

plt.figure(5)
plt.hist(KLD_vec)
plt.plot(KLD_vec, np.ones(KLD_vec.shape), 'rx')
plt.plot(np.ones(25) * np.median(KLD_vec), np.linspace(0, 25, 25), 'k-')
plt.plot(np.ones(25) * np.mean(KLD_vec), np.linspace(0, 25, 25), 'g-')
plt.xlim([0, 6])
plt.title('KLD')

# scio.savemat("Kernel3.mat", {'kernel': kernel})
plt.show()
