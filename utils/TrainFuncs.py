import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def diceLoss(pred, mask):
    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


def kernelEntropyCalc(Model):
    entropy_sum_3D = 0
    entropy_sum_2D = 0

    for i in range(64):
        input_noise = tf.random.uniform([1, 512, 512, 3, 1], -1, 1)

        for j in range(10):
            with tf.GradientTape() as entropy_tape:
                entropy_tape.watch(input_noise)
                _, dn5 = Model(input_noise)
                loss = tf.reduce_mean(dn5[:, :, :, :, i])
                grad = entropy_tape.gradient(loss, input_noise)
                grad_norm = grad / (tf.sqrt(tf.reduce_mean(tf.square(grad))) + 1e-5)
                input_noise += grad_norm

        kernel = np.ravel(input_noise.numpy())
        # kernel1 = np.ravel(input_noise.numpy()[:, :, 0])
        # kernel2 = np.ravel(input_noise.numpy()[:, :, 1])
        # kernel3 = np.ravel(input_noise.numpy()[:, :, 2])

        counts, _ = np.histogram(kernel, 64)
        prob = counts / np.sum(counts)
        info = np.log2(prob)

        entropy = -np.nansum(prob * info)
        entropy_sum_3D += entropy

        # counts, _ = np.histogram(kernel1, 64)
        # prob = counts / np.sum(counts)
        # info = np.log2(prob)
        #
        # entropy = -np.nansum(prob * info)
        # entropy_sum_2D += entropy
        #
        # counts, _ = np.histogram(kernel2, 64)
        # prob = counts / np.sum(counts)
        # info = np.log2(prob)
        #
        # entropy = -np.nansum(prob * info)
        # entropy_sum_2D += entropy
        #
        # counts, _ = np.histogram(kernel3, 64)
        # prob = counts / np.sum(counts)
        # info = np.log2(prob)
        #
        # entropy = -np.nansum(prob * info)
        # entropy_sum_2D += entropy

        del input_noise
        del kernel
        # del kernel1
        # del kernel2
        # del kernel3

    return entropy_sum_3D / 64, entropy_sum_2D / 64


def trainStep(imgs, segs, Model, ModelOptimiser):
    mean_entropy_3D, mean_entropy_2D = kernelEntropyCalc(Model)

    with tf.GradientTape() as tape:
        prediction, _ = Model(imgs, training=True)
        loss = diceLoss(prediction, segs)

    gradients = tape.gradient(loss, Model.trainable_variables)
    ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

    return loss, mean_entropy_3D, mean_entropy_2D


def valStep(imgs, labels, Model):
    prediction = Model(imgs, training=False)
    loss = diceLoss(prediction, labels)
    
    return loss


if __name__ == "__main__":
    noise = tf.random.normal((512, 512, 3), 0, 1)
    img = np.zeros((512, 512, 3))
    img[156:356, 156:356, :] = 1
    img = tf.convert_to_tensor(img)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(noise.numpy())
    axs[0, 1].imshow(img.numpy())
    axs[1, 0].hist(tf.reshape(noise, (1, -1)), 64)
    axs[1, 1].hist(tf.reshape(img, (1, -1)), 64)

    noise_counts, noise_bins = np.histogram(np.ravel(noise.numpy()), 64)
    img_counts, img_bins = np.histogram(np.ravel(img.numpy()), 64)
    noise_prob = noise_counts / np.sum(noise_counts)
    img_prob = img_counts / np.sum(img_counts)

    assert np.sum(noise_prob) == 1.0
    assert np.sum(img_prob) == 1.0

    noise_info = np.log2(noise_prob)
    img_info = np.log2(img_prob)

    noise_entropy = -np.nansum(noise_prob * noise_info)
    img_entropy = -np.nansum(img_prob * img_info)

    print(noise_entropy, img_entropy)

    plt.show()
