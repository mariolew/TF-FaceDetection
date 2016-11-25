import glob

filenames = glob.glob('data_prepare/net_positive/*.jpg')

lines = map(lambda x:x+' 1\n', filenames)

with open('net_12_pos_list.txt','w') as f:
    f.writelines(lines)


filenames = glob.glob('data_prepare/net_positive_flip/*.jpg')

lines = map(lambda x:x+' 1\n', filenames)

with open('net_pos_flip_list.txt','w') as f:
    f.writelines(lines)


filenames = glob.glob('data_prepare/net_negative/*.jpg')

lines = map(lambda x:x+' 0\n', filenames)

with open('net_12_neg_list.txt','w') as f:
    f.writelines(lines)