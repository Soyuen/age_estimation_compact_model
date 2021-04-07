import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import  Adam
from aec_model import *
from TYY_utils import mk_dir, load_data_npz
import TYY_callbacks
from generators import *
import cv2
import tensorflow as tf
import shutil
import random
import time
import math
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database npz file")
    parser.add_argument("--db", type=str, required=True,
                        help="imdb or wiki")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    args = parser.parse_args()
    return args

def main():
    tstart = time.time()
    args = get_args()
    input_path = args.input
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    logging.debug("Loading data...")
    image, age, image_size , age20 ,age4 , age1 = load_data_npz(input_path)
    x_data = image
    y_data_a = age
    y20_data = age20
    y4_data = age4
    y1_data = age1
    start_decay_epoch = [30,60]
    model = AEC_model((64,64,3))
    save_name = 'aec_model'
    if db_name == "imdb":
        model.compile(optimizer=Adam(), loss={'pred_a':'mae','pre_4':pred4_newloss,'pre_1':pred1_newloss,'pre_cod':cod},loss_weights={'pred_a':3,'pre_4':8.5,'pre_1':12,'pre_cod':18})
    else:
        model.compile(optimizer=Adam(), loss={'pred_a':'mae','pre_4':wpred4_newloss,'pre_1':wpred1_newloss,'pre_cod':cod},loss_weights={'pred_a':3,'pre_4':8.5,'pre_1':12,'pre_cod':18})
    if db_name == "wiki":
        weight_file = "imdb_models/weights.hdf5"
        model.load_weights(weight_file)
    
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_checkpoints")
    
    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_pred_a_loss:.3f}.hdf5",
                                 monitor="val_pred_a_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode="auto"), decaylearningrate
                        ]
    logging.debug("Running training...")
    
    data_num = len(x_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_data_a = y_data_a[indexes]
    y20_data = y20_data[indexes]
    y4_data = y4_data[indexes]
    y1_data = y1_data[indexes]    
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]
    y20_train = y20_data[:train_num]
    y4_train = y4_data[:train_num]
    y4_test = y4_data[train_num:]+y20_data[train_num:]
    y1_test = y_data_a[train_num:]
    
    hist = model.fit_generator(generator=data_generator_reg(X=x_train, Y=y_train_a,Y20=y20_train,Y4=y4_train,batch_size=batch_size),
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(x_test,{'pred_a': y_test_a,'pre_4':y4_test,'pre_1':y1_test,'pre_cod':y_test_a}),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
    
    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models", save_name+'.h5'), overwrite=True)
    model.save(os.path.join(db_name+"_models/"+'aec_model.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models", 'history_'+save_name+'.h5'), "history")
if __name__ == '__main__':
    main()