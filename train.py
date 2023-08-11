import os
import numpy as np
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


# with tf.device('/cpu:0'):
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_step = 25

model = keras.Sequential()
model.add(keras.layers.LSTM(units=64, input_shape=(time_step, 1), return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.LSTM(units=32, return_sequences=False))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.RepeatVector(n=time_step))

# model.add(keras.layers.LSTM(units=32, return_sequences=True))
# model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))
model.compile(loss='mae', optimizer='adam')



if __name__ == '__main__':
    input_path = 'C:\\Users\\pc\\Desktop\\Anomaly Detection\\datasets\\selected datasets'
    for file in os.listdir(input_path):
        # df = pd.read_csv(f'C:\\Users\\pc\\Downloads\\Compressed\\time-series\\{file}.csv')
        df = pd.read_csv(os.path.join(input_path, file))
        df = df[df.label == 0]
        
        
        print(f'####################On file {file}  ############################')
        
        scaler = StandardScaler()
        df['value'] = scaler.fit_transform(df[['value']])

        
        X_train, y_train = create_dataset(df[['value']], df.value,time_step)
        print('Start training........ ')
        history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.1,
        shuffle=False
        )

    model.save('./models/new_model.h5')
