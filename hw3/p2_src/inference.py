import os 
import sys 

data_dir, save_dir = sys.argv[1], sys.argv[2] 
data_names = os.listdir(data_dir)
save_names = [n.split('.')[0]+'.png' for n in data_names]
data_paths = [os.path.join(data_dir, n) for n in data_names]
save_paths = [os.path.join(save_dir, n) for n in save_names]
for d, s in zip(data_paths, save_paths):
    os.system(f'python3 ./p2_src/visualize.py {d} {s}')



