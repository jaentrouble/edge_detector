import tensorflow as tf
from edge_models import *
from model_trainer import EdgeModel
from tensorflow.keras import mixed_precision
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import os
from tools.stitch import *
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-w','--weight', dest='weight',required=True)
parser.add_argument('-n','--name', dest='name',required=True)
parser.add_argument('-i','--input',dest='input',required=True)
parser.add_argument('-o', '--output',dest='output', required=True)
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

patch_size = (320,320)
overlap = 30
model_f = ehrb0_112_12
weight_dir = args.weight
input_shape = [patch_size[1],patch_size[0],3]

edge_model = EdgeModel(input_shape, model_f)
edge_model.compile(
    optimizer='adam',
)
edge_model.load_weights(weight_dir)

vid_dir = Path(args.input)
edge_dir = Path(args.output)
vid_names = os.listdir(vid_dir)

for vid_name in vid_names:
    print(f'{vid_name} start')
    cap = cv2.VideoCapture(str(vid_dir/vid_name))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    frame_size = (frame.shape[1],frame.shape[0])
    frame_size_hw = (frame_size[1],frame_size[0])
    writer = cv2.VideoWriter(
        str(edge_dir/f'{os.path.splitext(vid_name)[0]}_edge_{args.name}.mp4'),
        fourcc,
        24,
        frame_size
    )

    print('Counting frames...')
    nb_original_frames = int(subprocess.run(
        [
            'ffprobe', 
            '-v', 
            'fatal', 
            '-count_frames',
            '-select_streams',
            'v:0',
            '-show_entries', 
            'stream=nb_read_frames', 
            '-of', 
            'default=noprint_wrappers=1:nokey=1', 
            str(vid_dir/vid_name)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT).stdout
    )
    print(f'Total {nb_original_frames} frames')
    t = tqdm(unit='frames',total=nb_original_frames,
             desc=f'Processing {vid_name}', leave=False)

    while cap.isOpened():
        if ret:
            f = frame.astype(np.float32) / 255.0
            patches = frame_to_patch(f, patch_size, overlap)
            edge_patches = edge_model.predict_on_batch(patches)[...,np.newaxis]
            edge_f = patch_to_frame(edge_patches, frame_size_hw, overlap)
            edge_f_uint8 = np.round(edge_f*[255,255,255])\
                            .astype(np.uint8)
            writer.write(edge_f_uint8)

        else:
            break

        ret, frame = cap.read()
        t.update()

    t.close()
    cap.release()
    writer.release()
    print(f'{vid_name} end')