from PIL import Image, ImageFilter
import os

def make_lr(hr_path, lr_path, scale=4):
    img = Image.open(hr_path)
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    w, h = img.size
    img_lr = img.resize((w//scale, h//scale), Image.BICUBIC)
    img_lr.save(lr_path)

label = 4
for image in os.listdir(f'/path/to/test_separados/{label}/img/'):
    make_lr(f'/path/to/test_separados/{label}/{image}', f'/path/to/test_separados/{label}/lr/{image}', scale=4)
