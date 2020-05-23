from __future__ import print_function, division
import sys
import glob 
import pandas as pd
import csv
import os

path_to_dataset = '/usr/home/kop/datasets/EgoGesture'

paths = sorted(glob.glob(os.path.join(path_to_dataset,'labels-final-revised1/*/*/*' )))
########################################################
##################### SUBJECT LIST #####################
########################################################
subject_ids = ['{num:02d}'.format(num=i) for i in range(1,51)]

subject_ids_train = ['{num:02d}'.format(num=i) for i in [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23,\
														 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44,\
														  45, 46, 48, 49, 50]]
subject_ids_val = ['{num:02d}'.format(num=i) for i in [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]]
subject_ids_test = ['{num:02d}'.format(num=i) for i in [ 2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]

########################################################

def create_trainlist( subset ,file_name, class_types = 'all'):
	
	
	folder1 = 'Color'
	folder2 = 'rgb'
	if subset == 'training': # Check subject id for validation/train split
		subjects_to_process = subject_ids_train
	elif subset == 'validation':
		subjects_to_process = subject_ids_val
	elif subset == 'testing':
		subjects_to_process = subject_ids_test
	else:
		raise(ValueError("Subset cannot be no other than training, validation, testing"))


	print("Preparing Lines")
	new_lines = []
	for path in paths:
		df = pd.read_csv(path,index_col = False, header = None)
		x = path.rsplit(os.sep,4)
		subject = x[2]
		if subject[-2:] in subjects_to_process:

			index = x[-1].split('.')[0][-1]
			folder_path = os.path.join(subject.title(), x[3],'{}'.format(folder1),\
				'{}'.format(folder2)+ index)
			

			full_path = os.path.join('/'+x[0],'images',folder_path)
			n_images = len(sorted(glob.glob(full_path + '/*')))
			df_val = df.values
			start = 1
			end = df_val[1,1] -1 
			len_lines = df_val.shape[0]
			for i in range(len_lines):
				line = df_val[i,:]
				if class_types == 'all':
					if (line[1] - start) >= 8:# Some action starts right away so I do not add None LABEL
						new_lines.append(folder_path + ' ' + str(84)+ ' ' + str(start)+ ' ' + str(line[1]-1))
					new_lines.append(folder_path + ' ' + str(line[0])+ ' ' + str(line[1])+ ' ' + str(line[2]))
				elif class_types == 'all_but_None':
					new_lines.append(folder_path + ' ' + str(line[0])+ ' ' + str(line[1])+ ' ' + str(line[2]))
				elif class_types == 'binary':
					if (line[1] - start) >= 8:# Some action starts right away so I do not add None LABEL
						new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(line[1]-1))
					new_lines.append(folder_path + ' ' + '2' + ' ' + str(line[1])+ ' ' + str(line[2]))
				
				start = line[2]+1
			if (n_images - start >8):
				# Class 84 is None(Non Gesture) class
				if class_types == 'all':
					new_lines.append(folder_path + ' ' + '84'+ ' ' + str(start)+ ' ' + str(n_images))
				elif class_types == 'binary':
					new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(n_images))
		else:
			continue

	print("Writing to the file ...")
	file_path = os.path.join('annotation_EgoGesture',file_name)
	with open(file_path, 'w') as myfile:
	    for new_line in new_lines:
	    	myfile.write(new_line)
	    	myfile.write('\n')
	print("Scuccesfully wrote file to:",file_path)

if __name__ == '__main__':
	# This file helps to index videos in the dataset by creating a .txt file where every line is a video clip
	# has the gesture
	# The format of each line is as following: <path to the folder> <class index> <start frame> <end frame>
    subset = sys.argv[1]
    file_name = sys.argv[2]
    class_types = sys.argv[3]
    create_trainlist(subset, file_name, class_types)

    # HOW TO RUN:
    # python ego_prepare.py training trainlistall.txt all
