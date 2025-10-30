#!/usr/bin/env bash
zip -r hw3.zip 'shared_autonomy.py' 'intent_inference.py' 'train_coil.py' 'train_il.py' 'train_ildist.py' 'policies'

# python -m behavior_cloning.train_il --scenario intersection --goal left --epochs 200 --lr 5e-3
# python behavior_cloning/train_il.py --scenario intersection --goal left --epochs 200 --lr 5e-3
# python play.py --scenario intersection 
# python play.py --scenario lanechange
