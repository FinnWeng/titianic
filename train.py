from preprocess import Preprocess
import pandas as pd
import csv
import numpy as np
import tensorflow as tf

from net.cls import Classification_Net



class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    config = AttrDict()
    config.shuffle_buffer = 2000
    config.batch_size = 32
    config.base_lr = 1e-3
    config.log_dir = "./tf_log/"
    config.model_path = './model/titanic_cls.ckpt'
    config.size_of_ds = 891
    config.steps_per_epoch = (config.size_of_ds//config.batch_size) * 10
    

    return config

def define_model_config():
    config = AttrDict()
    config.mlp_dim = 64
    config.layer_num = 8
    config.out_dim = 2
    return config


def ds_preprocess(item_x, item_y):
    x = item_x
    y = item_y
    one_hot_y = tf.one_hot(tf.cast(y, tf.int32), 2)
    return (x, one_hot_y)


if __name__ == "__main__":
    config = define_config()
    model_config = define_model_config()
    # with open('train.csv', newline='') as csvfile:

    #     rows = csv.DictReader(csvfile)


    #     for row in rows:
    #         print(row)

    train = pd.read_csv("./data/train.csv")
    test    = pd.read_csv("./data/test.csv")

    preprocess = Preprocess(config, train=train, test= test)
    train_x, train_y, test_x = preprocess.preprocess()

    # print(np.max(train_x)) (b, 16)
    # print(np.min(train_x))
    print("train_x:",train_x.shape) # (891, 16)

    # print(np.max(test_x))
    # print(np.min(test_x))
    print("test_X", test_x.shape) # (418, 16)

    # print(np.max(train_y))
    # print(np.min(train_y))
    # (b,)
    print("train_y:", train_y.shape)


    ds_train_x = tf.data.Dataset.from_tensor_slices(train_x)
    ds_train_y = tf.data.Dataset.from_tensor_slices(train_y)
    ds_train = tf.data.Dataset.zip((ds_train_x, ds_train_y))
    ds_train = ds_train.map(ds_preprocess,  tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(config.shuffle_buffer)
    ds_train = ds_train.batch(config.batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # for one_item in ds_train.take(100):
    #     print(one_item[1])
    
    model = Classification_Net(model_config.mlp_dim, model_config.layer_num, model_config.out_dim, "titanic_classification")



    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= config.log_dir, histogram_freq=1, update_freq = 100)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.model_path,
        save_weights_only= True,
        verbose=1)

    callback_list = [tensorboard_callback,save_model_callback]

    optimizer = tf.keras.optimizers.Adam(learning_rate = config.base_lr)

    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = tf.keras.metrics.CategoricalAccuracy()
        )

    # print(model.summary())


    hist = model.fit(ds_train,
            epochs=10, 
            steps_per_epoch=config.steps_per_epoch,
            callbacks = callback_list
            ).history
    
    last_predict = tf.argmax(tf.nn.softmax(model(test_x)), axis = -1).numpy()

    print(last_predict)
    print(last_predict.shape) # (418,)


