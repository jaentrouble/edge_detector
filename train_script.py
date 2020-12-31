import os

import numpy as np
from model_trainer import run_training
import edge_models
import model_lr
import argparse
import tensorflow as tf
import imageio as io
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model', required=True)
parser.add_argument('-lr', dest='lr', required=True)
parser.add_argument('-n','--name', dest='name',required=True)
parser.add_argument('-e','--epochs', dest='epochs', type=int)
parser.add_argument('-s','--steps', dest='steps', required=True, type=int)
parser.add_argument('-b','--batchsize', dest='batch', default=32, type=int)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
parser.add_argument('-pf','--profile', dest='profile',
                    action='store_true',default=False)
parser.add_argument('--load',dest='load',default=False)

args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

train_data_dir = Path('data/train')
val_data_dir = Path('data/val')

model_f = getattr(flow_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float
batch_size = int(args.batch)
profile = args.profile
steps_per_epoch = int(args.steps)
load_model_path = args.load

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = batch_size
kwargs['steps_per_epoch'] = steps_per_epoch
kwargs['train_data_dir'] = train_data_dir
kwargs['val_data_dir'] = val_data_dir
kwargs['image_size'] = (320,320)
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False
kwargs['profile'] = profile
kwargs['load_model_path'] = load_model_path

run_training(**kwargs)