import os
import numpy as np
from typing import List
'''
Split test set from train set.
Please modify it before running.
'''


## select all files in the root and generate a filelist txt
origin_folder = r"D:\syx-working\quant\stockdata\wukong"
out_folder = r"D:\syx-working\quant\stockdata\test"
# origin_folder = r"/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/wukong"
# out_folder = r"/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/test"

percent = 0.2

# processs
filelist = []
for root, _, names in os.walk(origin_folder):
	folder_prefix = root[len(origin_folder)+1:]
	for name in names:
		if name.endswith(".csv"):
			filename = folder_prefix + '/' + name if len(folder_prefix) > 0 else name
			filelist.append(filename)

total_length = len(filelist)
number = int(percent * total_length)
select_indices = np.random.choice(total_length, size=number, replace=False)

# get all files and save
selected_files = [filelist[idx] for idx in select_indices]
print("Total files :", total_length)
print("Selected files: ", len(selected_files))

## move files
for name in selected_files:
	in_file = os.path.join(origin_folder, name)
	out_file = os.path.join(out_folder, name)
	os.rename(in_file, out_file)
	print("Move file: ", name)


## save selected files to txt
def save_list(filelist: List[str], out_file: str):
	out_file = '../testset.txt'
	with open(out_file, mode='w', encoding='utf-8') as f:
		N = len(selected_files)
		for i in range(N-1):
			f.write(selected_files[i]+'\n')
		f.write(selected_files[N-1])

def move_files(namelist: List[str], in_folder: str, out_folder: str):
	for name in namelist:
		in_file = os.path.join(in_folder, name)
		out_file = os.path.join(out_folder, name)
		os.rename(in_file, out_file)
		print("Move file: ", name)

if __name__ == "__main__":

	print("End of the process.")