import glob
import random

filenames = glob.glob('data_prepare/net_positive/*.jpg')

lines1 = map(lambda x:x+' 1\n', filenames)


filenames = glob.glob('data_prepare/net_positive_flip/*.jpg')

lines2 = map(lambda x:x+' 1\n', filenames)


filenames = glob.glob('data_prepare/net_negative/*.jpg')

lines3 = map(lambda x:x+' 0\n', filenames)


lines = lines1 + lines2 + lines3
random.shuffle(lines)

with open('net_12_list.txt','w') as f:
	f.writelines(lines)
