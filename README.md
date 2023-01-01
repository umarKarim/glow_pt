This is a simple implementation of the Glow paper (https://arxiv.org/abs/1807.03039). This code is written for practice. I only use Additive coupling and do not use LU decomposition from the paper. The code at https://github.com/rosinality/glow-pytorch was quite helpful. 

## Training 
I used the CelebA training dataset available from https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ. Use the *img_align_celeba.zip* images and set path to the training data in the *main_train.py*.

To launch training, run 

`python3 main_train.py` 

Some snapshots from tensorboard.
https://github.com/umarKarim/cou_sfm/blob/main/figs/kitti_nyu_qual_crop.jpg
!(tensorboard_snapshots/losses.png)
!(tensorboard_snapshots/ims.png)


## Inference 
To use a trained model, run 

`python3 inference.py` 

You need to set the model path in the *inference.py* file. You can change the temprature inside the *inference.py* file. 
