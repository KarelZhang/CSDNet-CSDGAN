# CSDNet-CSDGAN

this is the code for the paper "Learning Deep Context-Sensitive Decomposition for Low-Light Image Enhancement"

## Environment Preparing
```
python 3.6
pytorch 0.4.1
```

### Testing

Download pretrained model from [Google drive](https://drive.google.com/drive/folders/1kocUbWn3aMkRX1_yeXNzY2bfCas07H6I?usp=sharing) 
or [Baidu drive](https://pan.baidu.com/s/1nTr3r72yUKLcTbSBwim0Cg) (extraction code:xcch). Then put them into `./checkpoints/`

Finally, run the script below, the results will be saved in `./results/`
```
python test.py 
--dataroot           #The folder path of the picture you want to test
E:/test/
--name               #The checkpoint name
CSDNet_UPE or CSDNet_LOL or CSDGAN or LiteCSDNet_UPE or LiteCSDNet_LOL or SLiteCSDNet_UPE or SLiteCSDNet_LOL
--gpu_ids            #could be single or multiple
0
```
A great thanks to [EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) for providing the basis for this code.
