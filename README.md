# Airbus Ship Detection Challenge

# Intro
The goal of this challenge is to develop a model that can accurately outline ships in satellite images. The dataset consists of satellite images along with corresponding ship masks. The ship masks are binary masks where white pixels represent ship regions and black pixels represent background regions.

# Key moments
1. Slightly custom Unet architecture
2. Using DataLoader "tf.keras.utils.Sequence"
3. Dice Loss & Dice coef. as metrics
4. Dataset balancing
5. Max simplicity


# How train & evaluate?
1. Clone repo
```shell
git clone https://github.com/ricottacake/airbus-ship-detection.git
```
2. Create virtual env
```shell
python3 -m venv .venv
source .venv/bin/activate
```
3. Install requirements
```shell
pip install -r requirements.txt
```
4. Train. In the config file, set the training directory ('train_v2' by default) as well as the file containing the mask for each snapshot ('train_ship_segmentations_v2.csv' by default)
```shell
python train.py
```
5. Evaluate. The result will be saved to the root folder
```shell
python inference.py image_name.jpg
```