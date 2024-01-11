import os
import pandas as pd

mydir = os.getcwd()

data_path = mydir + '\content'
cloud_path = data_path + '\\reArrange'
train_df = pd.read_csv(data_path+'\\train.csv')
test_df = pd.read_csv(data_path+'\\test.csv')