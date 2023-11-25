
import numpy as np
import tensorflow as tf


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient between the true and predicted values.

    Args:
        y_true (list): List of the true values for each image.
        y_pred (list): List of the predicted values for each image.
        smooth (float, optional): Smoothing factor to avoid division by zero, 1e-6 by default.

    Returns:
        tf.Tensor: The Dice coefficient between y_true and y_pred.
    """
    y_true_d = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_d = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_d * y_pred_d)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_d) + tf.reduce_sum(y_pred_d) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    """
    Computes the Dice loss between the true and predicted values.

    Args:
        y_true (list): List of the true values for each image.
        y_pred (list): List of the predicted values for each image.

    Returns:
        tf.Tensor: The Dice loss between y_true and y_pred.
    """
    return -dice_coef(y_true, y_pred)