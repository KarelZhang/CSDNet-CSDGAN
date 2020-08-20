import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import os
import torchvision.transforms as transforms


def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'unpair':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'pair':
        from data.pair_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'test':
        from data.test_dataset import TestDataset
        dataset = TestDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class TestDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'TestDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = TestDataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class TestDataset(BaseDataset):
    def __init__(self):
        super(TestDataset, self).__init__()

        self.root = 'G:\zja\data\DarkPair\LOL'
        self.dir_A = os.path.join(self.root, 'testA')
        self.dir_B = os.path.join(self.root, 'testB')

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        print(self.A_size)
        self.B_size = len(self.B_paths)

        self.transform = test_transform()

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
        A_gray = (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
        # A_gray = (1./A_gray)/255.
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'A_path':A_path, 'B_path':B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TestDataset'


def test_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]

    # transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)