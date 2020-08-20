import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = self.root

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)

        self.A_size = len(self.A_paths)

        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        A_img = self.A_imgs[index % self.A_size]
        A_path = self.A_paths[index % self.A_size]
        A_img = self.transform(A_img)
        r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
        A_gray = (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
        input_img = A_img
        return {'A': A_img, 'A_gray': A_gray, 'input_img': input_img, 'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'TestDataset'


