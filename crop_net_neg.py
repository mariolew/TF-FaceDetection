import os
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def random_crop(imagepath):

    image = Image.open(imagepath)
    len_rand = random.randint(0, min(image.size[0], image.size[1])-1)
    x_rand = random.randint(0, image.size[0]-len_rand-1)
    y_rand = random.randint(0, image.size[1]-len_rand-1)

    box = (x_rand, y_rand, x_rand+len_rand, y_rand+len_rand)
    return image.crop(box)


save_dir = 'data_prepare/net_negative'

if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)

neg_img_dir = 'imagenet_selected'
for file in os.walk(neg_img_dir):
    filenames = file[2]

for idx, filename in enumerate(filenames):

    filepath = neg_img_dir + '/' + filename

    for i in range(33):

        image_crop = random_crop(filepath)

        savepath = save_dir + '/' + str(i) + filename

        image_crop_12 = image_crop.resize((15,15))

        image_crop_12.save(savepath, 'jpeg')

    print idx
