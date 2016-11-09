import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sizes = (12, 24, 48)

for size in sizes:

    to_size = (size, size)

    with open('AFLW_ann.txt','r') as f:
        lines = f.readlines()

    save_dir = 'data_prepare/net_positive_' + str(size)

    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)

    for idx, line in enumerate(lines):
        s1 = line.strip().split(' ')
        image_path = s1[0]
        x = int(s1[1])
        y = int(s1[2])
        w = int(s1[3])
        h = int(s1[4])
        image = Image.open(image_path)
        box = (x, y, x+w, y+h)
        patch = image.crop(box)

        patch = patch.resize(to_size)

        s2 = image_path.split('/')
        image_name = s2[-1]

        save_path = save_dir+'/'+str(idx)+image_name

        patch.save(save_path, 'jpeg')