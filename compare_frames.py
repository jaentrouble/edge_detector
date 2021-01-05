import subprocess
import os
import argparse
from pathlib import Path
from multiprocessing import Pool, Queue, Process
from pprint import pprint
from tqdm import tqdm

SENTINEL = -1

def compare_framenum(input_paths, output_paths, Q):
    results = []
    for input_path, output_path in zip(input_paths, output_paths):
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
        vid_name = Path(input_path).name
        Q.put((vid_name, nb_input_frames==nb_output_frames))


def tqdm_counter(Q, final_Q, length):
    t = tqdm(total=length, unit='videos')
    result_list = []
    while True:
        result = Q.get()
        if result == SENTINEL:
            final_Q.put(result_list)
            t.close()
            return
        else:
            result_list.append(result)
            vid_name, is_same = result
            t.update()
            t.set_description(vid_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='input',required=True)
    parser.add_argument('-o', '--output',dest='output', required=True)
    parser.add_argument('-p', '--parallel',dest='parallel', required=True,type=int)
    args = parser.parse_args()


    vid_dir = Path(args.input)
    edge_dir = Path(args.output)
    vid_names = os.listdir(vid_dir)
    vid_names.sort()
    done_Q = Queue()
    final_Q = Queue()

    tqdm_proc = Process(target=tqdm_counter, 
                    args=(done_Q, final_Q, len(vid_names)))
    tqdm_proc.start()

    compare_procs = []
    per_worker = len(vid_names)//args.parallel
    for i in range(args.parallel):
        compare_procs.append(Process(
            target=compare_framenum,
            args=(
                [vid_dir/vn for vn in vid_names],
                [edge_dir/vn for vn in vid_names],
                done_Q,
            )
        ))
    for p in compare_procs:
        p.start()
    for p in compare_procs:
        p.join()

    done_Q.put(SENTINEL)
    tqdm_proc.join()
    results = final_Q.get()
    print('Not same vids:')
    for v, b in results:
        if not is_same:
            print(v)