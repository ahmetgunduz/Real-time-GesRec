import os
import cv2
import numpy as np
import glob
import sys
from subprocess import call

dataset_path = "/data2/nvGesture"
def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst',list_split = list()):
    file_with_split = os.path.join(dataset_path,file_with_split)
    params_dictionary = dict()
    with open(file_with_split,'rb') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.decode().split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1] 
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
                        
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split

def create_list(example_config, sensor,  class_types = 'all'):

    folder_path = example_config[sensor] + '_all'
    n_images = len(glob.glob(os.path.join(folder_path, '*.jpg')))

    label = example_config['label']+1
    start_frame = example_config[sensor+'_start']
    end_frame = example_config[sensor+'_end']

    
    frame_indices = np.array([[start_frame,end_frame]])
    len_lines = frame_indices.shape[0]
    start = 1
    for i in range(len_lines):
        line = frame_indices[i,:]
        if class_types == 'all':
            if (line[0] - start) >= 8:# Some action starts right away so I do not add None LABEL
                new_lines.append(folder_path + ' ' + '26'+ ' ' + str(start)+ ' ' + str(line[0]-1))
            new_lines.append(folder_path + ' ' + str(label)+ ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'all_but_None':
            new_lines.append(folder_path + ' ' + str(label)+ ' ' + str(line[0])+ ' ' + str(line[1]))
        elif class_types == 'binary':
            if (line[0] - start) >= 8:# Some action starts right away so I do not add None LABEL
                new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(line[0]-1))
            new_lines.append(folder_path + ' ' + '2' + ' ' + str(line[0])+ ' ' + str(line[1]))
        start = line[1]+1

    if (n_images - start >4):
        if class_types == 'all':
            new_lines.append(folder_path + ' ' + '26'+ ' ' + str(start)+ ' ' + str(n_images))
        elif class_types == 'binary':
            new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(n_images))

def extract_frames(sensors=["color", "depth"]):
    """Extract frames of .avi files.
    
    Parameters
    ----------
    modalities: list of str, ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    """
    for vt in sensors:
        files = glob.glob(os.path.join(dataset_path, 
                                       "Video_data",
                                       "*", "*", 
                                       "sk_" + vt + ".avi")) # this line should be updated according to the full path 
        for file in files:
            print("Extracting frames for ", file)
            directory = file.split(".")[0] + "_all"
            if not os.path.exists(directory):
                os.makedirs(directory)
            call(["ffmpeg", "-i",  file, os.path.join(directory, "%05d.jpg"), "-hide_banner"]) 
       
    
if __name__ == "__main__":
    sensors = ["color", "depth"]
    subset = sys.argv[1]
    file_name = sys.argv[2]
    class_types = sys.argv[3]

    sensors = ["color"]
    file_lists = dict()
    if subset == 'training':
        file_list = "./nvgesture_train_correct_cvpr2016_v2.lst"
    elif subset == 'validation':
        file_list = "./nvgesture_test_correct_cvpr2016_v2.lst"
    

    subset_list = list()

    load_split_nvgesture(file_with_split = file_list,list_split = subset_list)

    new_lines = [] 
    print("Processing Traing List")
    for sample_name in subset_list:
        create_list(example_config = sample_name, sensor = sensors[0], class_types = class_types)


    print("Writing to the file ...")
    file_path = os.path.join('annotation_nvGesture',file_name)
    with open(file_path, 'w') as myfile:
        for new_line in new_lines:
            myfile.write(new_line)
            myfile.write('\n')
    print("Scuccesfully wrote file to:",file_path)
    
    extract_frames(sensors=sensors)
