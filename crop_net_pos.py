import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open('AFLW_ann.txt','r') as f:
    lines = f.readlines()

save_dir1 = 'data_prepare/net_positive'
save_dir2 = 'data_prepare/net_positive_flip'

if os.path.exists(save_dir1)==False:
    os.makedirs(save_dir1)
if os.path.exists(save_dir2)==False:
    os.makedirs(save_dir2)

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

    patch1 = patch.resize((51, 51))
    patch2 = patch1.transpose(Image.FLIP_LEFT_RIGHT)

    s2 = image_path.split('/')
    image_name = s2[-1]

    save_path1 = save_dir1+'/'+str(idx)+image_name
    save_path2 = save_dir2+'/'+str(idx)+image_name

    patch1.save(save_path1, 'jpeg')
    patch2.save(save_path2, 'jpeg')

    print idx