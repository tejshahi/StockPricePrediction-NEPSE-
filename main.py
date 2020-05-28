
#import the necessary packages
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


#for filename
m_name=''
m_set=''

#1. Dataset with sentiment score
#2. Dataset without sentiment score(Only stock dataset)
dt_choice = input("Choose the dataset: ")

if(dt_choice == 1):
    df = read_csv('final__stock_news_data.csv', header=0, index_col=0)
    dataset = df[['LTP','Open','High','Low','Quantity','Score']]
    n_features = 6
    m_set = 'All' #for filename
else:
    df = read_csv('news_score_final.csv', header=0, index_col=0)
    dataset = df[['LTP','Open','High','Low','Quantity']]
    n_features = 5
    m_set = 'Only'

#1 LSTM 
#2 GRU
model_name = input("Enter the of the module(1/2): ")

dataset=dataset.reset_index()
dataset.head(2)

#set the date as index of dataset
dataset['Date']=pd.to_datetime(dataset.Date)
dataset.set_index('Date',drop=True,inplace=True)
#plot the closing price for trend analysis
plt.plot(dataset['LTP'])

dataset=dataset.sort_index()
dataset.head()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#prepare for the time-series generation
data=series_to_supervised(dataset, 12,1)
values = dataset.values
values = values.astype('float32')

#normalize the data using min-max normalizations
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of lag hours 10,12,14,16,18,20
n_lag = 12

# frame as supervised learning
reframed = series_to_supervised(scaled, n_lag, 1)

# split into train, validation and test sets
values = reframed.values
n_train = int(365 * 4)
n_val=int(n_train*0.8)
train = values[:n_val, :]
val=values[n_val:n_train,:]
test = values[n_train:, :]

#store the test data set index for comparing it with result
x_test_panda=ataset.iloc[n_train:,:]

# split into input and outputs
n_obs = n_lag * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
val_X, val_y=val[:,:n_obs],val[:,-n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = np.reshape(train_X, (train_X.shape[0], n_lag, n_features))
val_X = np.reshape(val_X, (val_X.shape[0], n_lag, n_features))
test_X = np.reshape(test_X, (test_X.shape[0], n_lag, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def plot_history(history):
    #plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def model_build(model_name):
    model = Sequential()
    if(model_name == 1):
        model.add(LSTM(120,input_shape=(train_X.shape[1], train_X.shape[2])))
        m_name = 'LSTM'
    else:
        model.add(GRU(120,input_shape=(train_X.shape[1], train_X.shape[2])))
        m_name = 'GRU'
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',metrics=['mse','mae'])
    return model

def da(y_true,y_pred):
    sum=0
    for i in range(y_true.size-1):
        dt=(y_true[i+1]-y_true[i])*(y_pred[i+1]-y_true[i])
        if(dt>=0):
            sum=sum+1
    return sum/(y_true.size-1)

def evaluateModel(inv_y,inv_yhat):
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    print('R2_Score: %.3f' % r2_score(inv_y,inv_yhat))
    print("MAE:", mean_absolute_error(inv_y,inv_yhat))
    print("Directionnal Accuracyc:", da(inv_y,inv_yhat))

model = model_build(model_name)

history = model.fit(train_X, train_y, epochs=100, batch_size=30, validation_data=(val_X, val_y),callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=2, shuffle=False)

plot_history(history)

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_lag*n_features))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

evaluateModel(inv_yhat,inv_y)

def trend_error(y_true,y_pred,n):
    s=type_one=type_two=0
    for i in range(y_true.size-1):
        actual_diff=y_true[i+1]-y_true[i]
        pred_diff=y_pred[i+1]-y_true[i]
        if(actual_diff<0 and pred_diff>0):
            type_one=type_one+1
        elif(actual_diff>0 and pred_diff<0):
            type_two=type_two+1
        else:
            s=s+1
    return (s/n,type_one/n,type_two/n)

trend_error(inv_y,inv_yhat, inv_y.size)

df = pd.DataFrame({'Actual':inv_y,'Predicted':inv_yhat,'Error': inv_y-inv_yhat,})

print(df.shape)
print(x_test_panda.shape)

y=x_test_panda.iloc[12:,]
true=y['LTP']

true_x=np.array(true)
pred_x=np.array(df['Actual'])

count=0
for i in range(true_x.size):
    if(true_x[i]==int(pred_x[i])):
        count=count+1
print(count)

output_data=df.set_index(y.index)

output_data.plot(subplots=True)
# save the output of each model to perform the DM-Test
file_name = m_name + '_' + m_set +'_Result.csv'
output_data.to_csv(file_name,index=True)
