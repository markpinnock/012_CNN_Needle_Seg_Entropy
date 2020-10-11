import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@tf.function
def dice_loss(pred, mask):

    """ Implements Dice loss
        - pred: predicted segmentation
        - mask: ground truth label """

    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


def kernel_entropy_calc(Model):

    """ Calculates kernel entropy of last layer of UNet decoder """

    # TODO: CHANGE TO ALLOW ANY NUMBER OF CHANNELS
    entropy_vec = np.zeros(64)

    for i in range(64):
        # TODO: CHANGE TO ALLOW ANY SIZE OF FEATURE MAPS
        # Input noise is the size of last layer of decoder
        input_noise = tf.random.uniform([1, 64, 64, 3, 1], -1, 1)

        # Perform gradient ascent to visualise kernel on feature map
        """ Understanding neural networks through deep visualisation """
        for j in range(5):
            with tf.GradientTape() as entropy_tape:
                entropy_tape.watch(input_noise)
                _, dn5 = Model(input_noise)
                loss = tf.reduce_mean(dn5[:, :, :, :, i])
                grad = entropy_tape.gradient(loss, input_noise)
                grad_norm = grad / (tf.sqrt(tf.reduce_mean(tf.square(grad))) + 1e-5)
                input_noise += grad_norm

        kernel = np.ravel(input_noise.numpy())

        # Calculate entropy of the kernel visualisation
        counts, _ = np.histogram(kernel, 64)
        prob = counts / np.sum(counts)
        prob = prob[np.nonzero(prob)]
        entropy = -np.sum(prob * np.log2(prob))
        entropy_vec[i] = entropy

    return entropy_vec


def trainStep(imgs, segs, Model, ModelOptimiser, lambd):

    """ Implements training step
        - imgs: input images
        - segs: segmentation labels
        - Model: model to be trained (keras.Model)
        - ModelOptimiser: e.g. keras.optimizers.Adam()
        - lambd: regularisation parameter (for kernel entropy) """

    # Calculate kernel entropy
    entropies = kernel_cntropy_calc(Model)

    # TODO: subclass UNet and convert train_step to class method
    with tf.GradientTape() as tape:
        prediction, _ = Model(imgs, training=True)
        loss = dice_loss(prediction, segs) + (lambd * np.median(entropies))

    gradients = tape.gradient(loss, Model.trainable_variables)
    ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

    return loss, entropies


@tf.function
def val_step(imgs, labels, Model):
    """ Implements validation step """

    # TODO: subclass UNet and convert val_step to class method
    prediction, _ = Model(imgs, training=False)
    loss = dice_loss(prediction, labels)
    
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

    # noise_counts, noise_bins = np.histogram(np.ravel(noise.numpy()), 64)
    # img_counts, img_bins = np.histogram(np.ravel(img.numpy()), 64)
    # noise_prob = noise_counts / np.sum(noise_counts)
    # img_prob = img_counts / np.sum(img_counts)
    # noise_prob = noise_prob[np.nonzero(noise_prob)]
    # img_prob = img_prob[np.nonzero(img_prob)]
    #
    # assert np.sum(noise_prob) == 1.0, print(np.sum(noise_prob))
    # assert np.sum(img_prob) == 1.0, np.sum(img_prob)
    #
    # noise_info = np.log2(noise_prob)
    # img_info = np.log2(img_prob)
    #
    # noise_entropy = -np.sum(noise_prob * noise_info)
    # img_entropy = -np.sum(img_prob * img_info)

    noise_idx = tf.histogram_fixed_width_bins(tf.reshape(noise, [-1]), [tf.reduce_min(noise), tf.reduce_max(noise)], 64)
    img_idx = tf.histogram_fixed_width_bins(tf.reshape(img, [-1]), [tf.reduce_min(img), tf.reduce_max(img)], 64)
    noise_idx = tf.sort(noise_idx)
    img_idx = tf.sort(img_idx)
    _, _, noise_counts = tf.unique_with_counts(noise_idx)
    _, _, img_counts = tf.unique_with_counts(img_idx)
    noise_prob = noise_counts / tf.reduce_sum(noise_counts)
    img_prob = img_counts / tf.reduce_sum(img_counts)

    assert tf.reduce_sum(noise_prob).numpy() == 1.0, tf.reduce_sum(noise_prob)
    assert tf.reduce_sum(img_prob).numpy() == 1.0, tf.reduce_sum(img_prob)

    noise_entropy = -tf.reduce_sum(noise_prob * tf.math.log(noise_prob))
    img_entropy = -tf.reduce_sum(img_prob * tf.math.log(img_prob))

    print(noise_entropy.numpy(), img_entropy.numpy())

    plt.show()
