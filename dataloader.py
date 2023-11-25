
import os
import math
import numpy as np

from skimage.io import imread
from keras.utils import Sequence
from utils import masks_as_image
from config import TRAIN_DIR


class ShipsDataset(Sequence):
    """
    Custom Keras Sequence class for loading ship segmentation data in batches.

    Parameters:
    - batch_size (int): The batch size for loading data in each iteration.
    - data (pd.DataFrame): DataFrame containing information about ship images and their corresponding masks.

    Attributes:
    - X (list): List of unique image IDs.
    - y (list): List of encoded mask information for each image.
    - batch_size (int): The batch size for loading data in each iteration.

    Methods:
    - __len__(): Returns the number of batches in the dataset.
    - __getitem__(idx): Generates one batch of data, loading images and their corresponding masks.

    Usage:
    dataset = ShipsDataset(batch_size=32, data=train_data)
    model.fit(dataset, epochs=10)
    """

    def __init__(self, batch_size, data):
        """
        Initializes the ShipsDataset instance.

        Parameters:
        - batch_size (int): The batch size for loading data in each iteration.
        - data (pd.DataFrame): DataFrame containing information about ship images and their corresponding masks.
        """
        data = data.groupby('ImageId')
        self.X = [x for x, y in data]
        self.y = [y['EncodedPixels'].values for x, y in data]
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        Returns:
        - int: The number of batches.
        """
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        """
        Generates one batch of data, loading images and their corresponding masks.

        Parameters:
        - idx (int): Index of the batch.

        Returns:
        - list: A list containing input images and target masks as NumPy arrays.
        """
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.X))

        batch_X = np.stack([imread(os.path.join(TRAIN_DIR, img))/255 for img in self.X[low:high]], 0)
        batch_y = np.stack([np.expand_dims(masks_as_image(mask), -1) for mask in self.y[low:high]], 0)

        return [batch_X, batch_y]
