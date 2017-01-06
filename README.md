# TF-FaceDetection
Reproduction of CVPR 15 paper of face detection, still ongoing.

# Usage
git clone https://github.com/mariolew/TF-FaceDetection

## Prepare data
cd data_utils
python get_ann.py
python generate_cal_list(net_list).py
python crop_cal(net_neg, net_pos).py

## Train
python train_net_12(24, 48).py
python train_cal_12(24, 48).py

## Test
python run.py
