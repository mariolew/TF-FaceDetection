import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_for_cal(sn, xn, yn, n):

    with open('AFLW_ann.txt','r') as f:
        lines = f.readlines()

    save_dir1 = 'data_prepare/cal_positive_'+str(n)+'_12'
    save_dir2 = 'data_prepare/cal_positive_'+str(n)+'_24'
    save_dir3 = 'data_prepare/cal_positive_'+str(n)+'_48'

    if os.path.exists(save_dir1)==False:
        os.makedirs(save_dir1)
    if os.path.exists(save_dir2)==False:
        os.makedirs(save_dir2)
    if os.path.exists(save_dir3)==False:
        os.makedirs(save_dir3)

    for idx, line in enumerate(lines):
        spl1 = line.strip().split(' ')
        image_path = spl1[0]
        x = int(spl1[1])
        y = int(spl1[2])
        w = int(spl1[3])
        h = int(spl1[4])

        x = int(x-xn*w/sn)
        y = int(y-yn*h/sn)
        w = int(w/sn)
        h = int(h/sn)

        image = Image.open(image_path)
        box = (x, y, x+w, y+h)
        patch = image.crop(box)

        patch1 = patch.resize((12, 12))
        patch2 = patch.resize((24, 24))
        patch3 = patch.resize((48, 48))

        spl2 = image_path.split('/')
        image_name = spl2[-1]

        save_path1 = save_dir1+'/'+str(idx)+image_name
        save_path2 = save_dir2+'/'+str(idx)+image_name
        save_path3 = save_dir3+'/'+str(idx)+image_name

        patch1.save(save_path1, 'jpeg')
        patch2.save(save_path2, 'jpeg')
        patch3.save(save_path3, 'jpeg')


if __name__ == '__main__':
    
    s_set = (0.83, 0.91, 1.0, 1.10, 1.21)
    x_set = (-0.17, 0, 0.17)
    y_set = (-0.17, 0, 0.17)
    n = 0
    for x in x_set:
        for y in y_set:
            for s in s_set:
                n = n + 1
                crop_for_cal(s, x, y, n)