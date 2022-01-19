import pandas as pd
from os.path import join


def define_label(data_dir):
    """
        Description:
            Define the label of each classes.
        Reture:
            (Tuple, Tuple)
    """    
    class_to_label = {}
    cur_valid_label = 0
    train_csv = join(data_dir, 'train.csv')
    train_dir = join(data_dir, 'train')
    train_df = pd.read_csv(train_csv).set_index('id')
    train_paths, train_labels = [],[]
    for i in range(len(train_df)):
        path = join(train_dir, train_df.loc[i, 'filename'])
        class_name = train_df.loc[i, 'label'] 
        if class_name not in class_to_label:
            class_to_label[class_name] = cur_valid_label
            label = class_to_label[class_name]
            cur_valid_label += 1 
        else:
            label = class_to_label[class_name]
        train_paths.append(path)
        train_labels.append(label)

    val_csv = join(data_dir, 'val.csv')
    val_dir = join(data_dir, 'val')
    val_df = pd.read_csv(val_csv).set_index('id')
    val_paths, val_labels = [],[]
    for i in range(len(val_df)):
        path = join(val_dir, val_df.loc[i, 'filename'])
        class_name = val_df.loc[i, 'label'] 
        if class_name not in class_to_label:
            class_to_label[class_name] = cur_valid_label
            label = class_to_label[class_name]
            cur_valid_label += 1 
        else:
            label = class_to_label[class_name]
        val_paths.append(path)
        val_labels.append(label)

    return (train_paths, train_labels), (val_paths, val_labels)

def load_eval_data(data_dir, data_csv):
    """
        Description:
            Load evaluation data
        Reture:
            (Tuple, Tuple)
    """    
    val_df = pd.read_csv(data_csv).set_index('id')
    val_paths, val_names = [], []
    for i in range(len(val_df)):
        path = join(data_dir, val_df.loc[i, 'filename'])
        val_paths.append(path)
        val_names.append(val_df.loc[i, 'filename'])
    return val_paths, val_names