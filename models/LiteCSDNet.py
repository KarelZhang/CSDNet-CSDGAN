import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


class LiteCSDNet(nn.Module):
    def __init__(self):
        super().__init__()

        p = 1
        self.channels = 12

        self.attention_conv1_1 = nn.Conv2d(2, self.channels, 3, padding=p)
        self.attention_LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn1_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv1_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn1_2 = nn.BatchNorm2d(self.channels)
        self.attention_max_pool1 = nn.MaxPool2d(2)

        self.attention_conv2_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn2_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv2_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn2_2 = nn.BatchNorm2d(self.channels)
        self.attention_max_pool2 = nn.MaxPool2d(2)

        self.attention_conv3_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn3_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv3_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn3_2 = nn.BatchNorm2d(self.channels)
        self.attention_max_pool3 = nn.MaxPool2d(2)

        self.attention_conv4_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn4_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv4_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn4_2 = nn.BatchNorm2d(self.channels)

        self.attention_deconv6 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_conv7_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.attention_LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn7_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv7_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn7_2 = nn.BatchNorm2d(self.channels)

        self.attention_deconv7 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_conv8_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.attention_LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn8_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv8_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn8_2 = nn.BatchNorm2d(self.channels)

        self.attention_deconv8 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_conv9_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.attention_LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn9_1 = nn.BatchNorm2d(self.channels)
        self.attention_conv9_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.attention_LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_conv10 = nn.Conv2d(self.channels, 1, 3, padding=p)
        self.attention_sigmoid_5 = nn.Sigmoid()
        self.attention_sigmoid_4 = nn.Sigmoid()
        self.attention_sigmoid_3 = nn.Sigmoid()
        self.attention_sigmoid_2 = nn.Sigmoid()
        self.attention_sigmoid_1 = nn.Sigmoid()
        self.attention_sigmoid = nn.Sigmoid()

        #down
        self.conv1_1 = nn.Conv2d(3, self.channels, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(self.channels)
        self.conv1_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(self.channels)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(self.channels)
        self.conv2_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(self.channels)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(self.channels)
        self.conv3_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(self.channels)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(self.channels)
        self.conv4_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(self.channels)

        self.deconv6 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.conv7_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(self.channels)
        self.conv7_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(self.channels)

        self.deconv7 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.conv8_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(self.channels)
        self.conv8_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(self.channels)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.conv9_1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(self.channels)
        self.conv9_2 = nn.Conv2d(self.channels, self.channels, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv10 = nn.Conv2d(self.channels, 3, 3, padding=p)
        self.tanh = nn.Tanh()

    def forward(self, input, gray):
        input_gray = gray
        flag = 0
        self.original_features = []
        self.divided_features = []
        self.illu_features = []
        self.conv1_features = []
        self.conv2_features = []
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)

        conv1 = self.attention_bn1_1(self.attention_LReLU1_1(self.attention_conv1_1(gray)))
        x = self.attention_max_pool1(conv1)

        conv2 = self.attention_bn2_1(self.attention_LReLU2_1(self.attention_conv2_1(x)))
        x = self.attention_max_pool2(conv2)

        conv3 = self.attention_bn3_1(self.attention_LReLU3_1(self.attention_conv3_1(x)))
        x = self.attention_max_pool3(conv3)

        x = self.attention_bn4_1(self.attention_LReLU4_1(self.attention_conv4_1(x)))
        conv4 = self.attention_bn4_2(self.attention_LReLU4_2(self.attention_conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')
        up7 = torch.cat([self.attention_deconv6(conv6), conv3], 1)
        x = self.attention_bn7_1(self.attention_LReLU7_1(self.attention_conv7_1(up7)))
        conv7 = self.attention_bn7_2(self.attention_LReLU7_2(self.attention_conv7_2(x)))
        self.gray_3 = self.attention_sigmoid_3(conv7)

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        up8 = torch.cat([self.attention_deconv7(conv7), conv2], 1)
        x = self.attention_bn8_1(self.attention_LReLU8_1(self.attention_conv8_1(up8)))
        conv8 = self.attention_bn8_2(self.attention_LReLU8_2(self.attention_conv8_2(x)))
        self.gray_2 = self.attention_sigmoid_2(conv8)

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        up9 = torch.cat([self.attention_deconv8(conv8), conv1], 1)
        x = self.attention_bn9_1(self.attention_LReLU9_1(self.attention_conv9_1(up9)))
        conv9 = self.attention_LReLU9_2(self.attention_conv9_2(x))
        self.gray_1 = self.attention_sigmoid_1(conv9)

        latent = self.attention_conv10(conv9)
        self.gray = self.attention_sigmoid(latent)

        conv1 = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        x = self.max_pool1(conv1)

        conv2 = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        x = self.max_pool2(conv2)

        conv3 = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')
        conv6 = self.deconv6(conv6)
        conv6 = conv6 / self.gray_3
        conv3 = conv3 / self.gray_3
        up7 = torch.cat([conv6, conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))
        self.conv2_features.append(conv7)

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        conv7 = self.deconv7(conv7)
        conv7 = conv7 / self.gray_2
        conv2 = conv2 / self.gray_2
        up8 = torch.cat([conv7, conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))
        self.conv2_features.append(conv8)

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        conv8 = self.deconv8(conv8)
        conv8 = conv8 / self.gray_1
        conv1 = conv1 / self.gray_1
        up9 = torch.cat([conv8, conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))
        self.conv2_features.append(conv9)

        latent = self.tanh(self.conv10(conv9))

        output = latent / self.gray

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(self.gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')

        return output, latent, gray








