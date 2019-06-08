from time import time
# Starts the timer
start = time()
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os

def is_valid_file(parser, arg):
    # Checks if the file exists
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  

parser = argparse.ArgumentParser(description='''
                                    ANN regressor built in tensorflow to predict the value of y based on 304 features.  
                                    prints the MSE of the model and generated the file predictions.csv with the label and the prediction
                                    ''')
parser.add_argument("-i", dest="file", required=True,
                    help="input file with 305 columns including y", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
args = parser.parse_args()

file = args.file

try:
    df = pd.read_csv(file, low_memory = False)
    # Eliminate columns with many NANs
    na_cols = pd.DataFrame(df.isnull().sum().sort_values())
    na_cols.reset_index(drop = False, inplace = True)
    na_cols.columns = (['feature','num_NaN'])
    na_cols.loc[:,'tenpct'] = np.where(na_cols.num_NaN > 10000, 'Y', 'N')
    df1 = df[df.columns[df.columns.isin(na_cols[na_cols.tenpct == 'Y'].feature.unique()) == False]].dropna().reset_index(drop = True)
    # Eliminate columns with only one or two observations
    dic1= {}
    dic2= {}
    dic_= {}
    for column in df1.columns:
        unique_ = df1[column].nunique()
        if unique_ == 1:
            dic1[column] = [unique_]
        if unique_ == 2:
            dic2[column] = [unique_]
        else:
            dic_[column] = [unique_]
    df2 = df1[df1.columns[df1.columns.isin(dic1.keys()) == False]].dropna().reset_index(drop = True)
    # Scale the values on each column to MIN MAX scale for each feature
    from sklearn.preprocessing import MinMaxScaler
    df2n= df2.copy()
    for column in df2n.columns[df2n.columns != 'y']:
        df2n.loc[:, column ] = MinMaxScaler().fit_transform(df2n[column].astype(np.float).values.reshape(-1,1))
    # Remove the labels (column (y)
    running_set = df2n.copy()
    running_labels = running_set.pop('y')    
    # Load the model
    new_model = keras.models.load_model('prediction_model.h5')
    # Evaluate the model
    loss, mae, mse = new_model.evaluate(running_set, running_labels, verbose = 0)
    print("Testing set Mean Abs Error: {:5.2f}".format(mse))
    # Generate predictions
    predictions = new_model.predict(running_set).flatten()
    predictions = pd.Series(predictions).to_frame().rename(columns = {0:'predictions'})
    # Saves the prediction with respective labels
    pd.Series(running_labels).to_frame().merge(predictions, left_index=True, right_index=True).to_csv('predictions.csv', index = False, header = True)
    # Ends timer and prints time of execution
    end = time()
    print('End to end, getting the predictions required {} minutes'.format((end - start)/60))
except:
    print ("There was an error with the code.")
    sys.exit()

