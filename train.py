from preprocess import Preprocess
import pandas as pd
import csv
import numpy as np



class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    config = AttrDict()
    # config.tfr_fname =  "/home/data_storage1/handwritting_merge/img_clf.tfrecord"
    # config.tfr_fname =  "/home/data_storage1/handwritting_merge/origin_img_clf.tfrecord"
    config.tfr_fname =  "/home/data_storage1/handwritting_merge/img_clf_rnn.tfrecord"
    # config.tfr_fname =  "/home/data_storage1/handwritting_merge/img_clf_rnn_try.tfrecord"
    # config.tfr_fname =  "./img_clf.tfrecord"
    config.cls_label_path = "./total_cls_labels.txt"
    config.shuffle_buffer = 10000
    config.batch_size = 180
    config.require_size = [224,224]
    

    return config


if __name__ == "__main__":
    config = define_config()
    # with open('train.csv', newline='') as csvfile:

    #     rows = csv.DictReader(csvfile)


    #     for row in rows:
    #         print(row)

    train = pd.read_csv("./data/train.csv")
    test    = pd.read_csv("./data/test.csv")

    preprocess = Preprocess(config, train=train, test= test)
    train_x, train_y, test_x = preprocess.preprocess()

    print(np.max(train_x))
    print(np.min(train_x))

    print(np.max(test_x))
    print(np.min(test_x))
