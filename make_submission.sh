#!/usr/bin/env bash
zip -r hw3.zip 'shared_autonomy.py' 'intent_inference.py' 'train_coil.py' 'train_il.py' 'train_ildist.py' 'policies'


# python -m behavior_cloning.train_il --scenario intersection --goal left --epochs 400 --lr 1e-3 [Default MSE]
# python -m behavior_cloning.train_il --scenario intersection --goal left --loss l1 --input_noise_std 0.005 --early_stop --patience 20 --min_delta 5e-4 --epochs 400 --lr 7e-4
# python -m behavior_cloning.train_il   --scenario intersection --goal left   --loss huber --huber_delta 0.25   --input_noise_std 0.005   --early_stop --patience 20 --min_delta 5e-4   --epochs 400 --lr 7e-4
# python -m behavior_cloning.train_il   --scenario intersection --goal left   --loss mix --mix_alpha 0.7   --w_steer 1.2 --w_throttle 1.0   --input_noise_std 0.005   --early_stop --patience 20 --min_delta 5e-4   --epochs 400 --lr 7e-4


# python -m behavior_cloning.test_il --scenario intersection --goal left --visualize
# python -m behavior_cloning.test_il --scenario intersection --goal left

# python play.py --scenario intersection 
# python play.py --scenario lanechange
