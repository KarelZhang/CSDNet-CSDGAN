import time
import os
import time
import torch
import ntpath
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
# from pdb import set_trace as st
# from util import html
from models.test_model import TestModel
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = TestModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))
# visualizer = Visualizer(opt)
# create website
out_dir = os.path.join("./ablation/", opt.name)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print('the number of test image:')
print(len(dataset))


def save_images(path, visuals, image_path):
    image_dir = path
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, image_numpy in visuals.items():
        if label is 'fake_B':
            image_name = '%s.png' % name
            #image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            save_image(image_numpy, save_path)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


with torch.no_grad():
    for i, data in enumerate(dataset):
        model.set_input(data)
        start_time = time.time()
        visuals = model.predict()
        end_time = time.time()
        print('using %f seconds' % (end_time - start_time))
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        save_images(out_dir, visuals, img_path)


