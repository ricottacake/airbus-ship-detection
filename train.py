
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from model import get_model
from utils import dice_coef, dice_loss
from dataloader import ShipsDataset
from config import EPOCHS, BATCH_SIZE, SAMPLES_PER_GROUP, MASK_FILE


masks = pd.read_csv(MASK_FILE)
masks['ships'] = masks['EncodedPixels'].map(lambda row: 1 if isinstance(row, str) else 0)
unique_images = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
masks.drop(['ships'], axis=1, inplace=True)
balanced_train_df = unique_images.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

train_ids, valid_ids = train_test_split(balanced_train_df, 
                                        test_size = 0.2,
                                        stratify = balanced_train_df['ships'])

train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)


weight_path="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(weight_path,
                             monitor='val_loss',
                             save_weights_only=True,
                             verbose=1,
                             mode='min')

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.33,
                                   patience=1,
                                   verbose=1,
                                   mode='min',
                                   min_delta=0.01,
                                   cooldown=0,
                                   min_lr=1e-8)

early = EarlyStopping(monitor="val_loss",
                      min_delta=0.01,
                      mode="min",
                      verbose=2,
                      patience=20)

callbacks_list = [checkpoint, early, reduceLROnPlat]

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

adam = Adam(learning_rate=lr_schedule)
model = get_model()
model.compile(optimizer=adam, loss=dice_loss, metrics=[dice_coef])

history = model.fit(ShipsDataset(BATCH_SIZE, train_df),
          validation_data=ShipsDataset(BATCH_SIZE, valid_df),
          epochs=EPOCHS,
          callbacks=callbacks_list)