import subprocess
import os
import argparse
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',dest='input',required=True)
parser.add_argument('-o', '--output',dest='output', required=True)
parser.add_argument('-p', '--parallel',dest='parallel', required=True,type=int)
args = parser.parse_args()

def compare_framenum(input_path, output_path):
    nb_input_frames = int(subprocess.run(
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
            str(input_path)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT).stdout
    )
    nb_output_frames = int(subprocess.run(
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
            str(output_path)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT).stdout
    )
    return nb_input_frames == nb_output_frames

vid_dir = Path(args.input)
edge_dir = Path(args.output)
vid_names = os.listdir(vid_dir)
vid_names.sort()

with Pool(processes=args.parallel) as pool:
    is_same = pool.starmap(compare_framenum, 
                        zip([vid_dir/v for v in vid_names],
                            [edge_dir/v for v in vid_names])

for v, b in zip(vid_names, is_same):
    if not is_same:
        print(v)