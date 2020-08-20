# CSDNet-CSDGAN

[this is the code for the paper "Learning Deep Context-Sensitive Decomposition for Low-Light Image Enhancement"]

## Environment Preparing
```
python 3.6
pytorch 0.4.1
```

### Testing

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put them into `./checkpoints/`

```
python test.py 
--dataroot           #The folder path of the picture you want to test
E:/test/
--name               #The checkpoint name
CSDNet_UPE or CSDNet_LOL or CSDGAN or LiteCSDNet_UPE
--dataset_mode
test
--network_model      #"normal" for CSDNet_UPE or CSDNet_LOL or CSDGAN, "lite" for LiteCSDNet_UPE
normal or lite
--gpu_ids            #could be single or multiple
0
```
