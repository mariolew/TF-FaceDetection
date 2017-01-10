# TF-FaceDetection
Reproduction of CVPR 15 paper of face detection, still ongoing.


# Prerequisite
Python 3.4

Tensorflow>=0.11.0


# Usage
git clone https://github.com/mariolew/TF-FaceDetection


## Prepare data
cd data_utils

python3 get_ann.py

python3 crop_cal(net_neg, net_pos).py

python3 generate_cal_list(net_list).py


## Train
python3 train_net_12.py

python3 net_12_eval.py(to determine threshold)

python3 net_24_neg.py

python3 train_net_24.py

python3 net_24_eval.py(to determine threshold)

python3 net_48_neg.py

python3 train_net_48.py

python3 net_48_eval.py(to determine threshold)

python3 train_cal_12(24, 48).py


## Test
python3 run.py

# Result
![demo](https://github.com/mariolew/TF-FaceDetection/raw/master/images/demo.png)