# this modul import the dm_test package which is available at https://github.com/johntwk/Diebold-Mariano-Test

from dm_test import dm_test
import pandas as pd

#read alll result files form stock data folder
gru_only=pd.read_csv('GRU-Only-Result.csv')

lstm_only=pd.read_csv('LSTM-Only-Result.csv')

gru_all=pd.read_csv('GRU-all-Result.csv')

lstm_all=pd.read_csv('LSTM-all-Result.csv')

from matplotlib import pyplot
pyplot.figure(figsize=(15,6), dpi=600)
pyplot.plot(gru_only['Actual'], label='Actual Price')
pyplot.plot(gru_only['Predicted'], label='Predicted Price')
pyplot.legend()
pyplot.xlabel('Date')
pyplot.ylabel("Price")
pyplot.grid(True)
pyplot.show()

def DM_test(df1,df2):
    actual=df1['Actual']
    first=df1['Predicted']
    second=df2['Predicted']
    actual_lst=np.array(actual)
    pred1_lst=np.array(first)
    pred2_lst=np.array(second)
    rtMSE=dm_test(actual_lst,pred1_lst,pred2_lst, h=1,crit='MSE')
    rtMAE=dm_test(actual_lst,pred1_lst, pred2_lst,h=1,crit='MAD')
    print("MSE value", rtMSE)
    print("MAE value", rtMAE)

import numpy as np
print("DM_Value for GRU-only and GRU-ALL")
DM_test(gru_only,gru_all)

print("DM value for LSTM-only and LSTM-ALl")
DM_test(lstm_only,lstm_all)

print("DM value for LSTM-only and GRU-All")
DM_test(lstm_only,gru_all)

print("DM value for GRU-only and LSTm ALL")
DM_test(gru_only,lstm_all)

print("DM value for GRU-All and LSTM_only")
DM_test(gru_all,lstm_only)

print("DM_value for GRU-Only and LSTM only")
DM_test(gru_only,lstm_only)

print("DM_Value for LSTM-All and GRU-ALL")
DM_test(gru_all,lstm_all)
