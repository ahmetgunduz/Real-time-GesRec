from __future__ import print_function, division
import os
import sys
import json
import pandas as pd

def convert_csv_to_dict(csv_path, subset, labels):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    key_start_frame = []
    key_end_frame = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        class_name = labels[row[1]-1]
        
        basename = str(row[0])
        start_frame = str(row[2])
        end_frame = str(row[3])

        keys.append(basename)
        key_labels.append(class_name)
        key_start_frame.append(start_frame)
        key_end_frame.append(end_frame)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]  
        if key in database: # need this because I have the same folder 3  times
            key = key + '^' + str(i) 
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        start_frame = key_start_frame[i]
        end_frame = key_end_frame[i]

        database[key]['annotations'] = {'label': label, 'start_frame':start_frame, 'end_frame':end_frame}
    
    return database

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(str(data.ix[i, 1]))
    return labels

def convert_nv_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training', labels)
    val_database = convert_csv_to_dict(val_csv_path, 'validation', labels)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    for class_type in ['all', 'all_but_None', 'binary']:

        if class_type == 'all':
            class_ind_file = 'classIndAll.txt'
        elif class_type == 'all_but_None':
            class_ind_file = 'classIndAllbutNone.txt'
        elif class_type == 'binary':
            class_ind_file = 'classIndBinary.txt'


        label_csv_path = os.path.join(csv_dir_path, class_ind_file)
        train_csv_path = os.path.join(csv_dir_path, 'trainlist'+ class_type + '.txt')
        val_csv_path = os.path.join(csv_dir_path, 'vallist'+ class_type + '.txt')
        dst_json_path = os.path.join(csv_dir_path, 'nv' + class_type + '.json')

        convert_nv_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, dst_json_path)
        print('Successfully wrote to json : ', dst_json_path)
    # HOW TO RUN:
    # python nv_json.py '../annotation_nvGesture'
