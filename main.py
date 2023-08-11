import os
import numpy as np
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# with tf.device('/cpu:0'):

# input_path = os.environ.get('DATA_104_PATH') + '/test_data'
input_path  = './time-series'
output_path = './output'


def f1_report(df):
    print(f'f1_score: {f1_score(df.label,df.label2)}')
    return f1_score(df.label,df.label2)

class TimeSeriesAnomalyDetector:
    def __init__(self,):
        pass
    
    def build_model(self,X_train):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.2))

        model.add(keras.layers.LSTM(units=32, return_sequences=False))
        model.add(keras.layers.Dropout(rate=0.2))

        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))

        # model.add(keras.layers.LSTM(units=32, return_sequences=True))
        # model.add(keras.layers.Dropout(rate=0.2))

        model.add(keras.layers.LSTM(units=64, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.2))

        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
        model.compile(loss='mae', optimizer='adam')

        return model

    def create_dataset(self,X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)



    def __call__(self, df):
        scaler = StandardScaler()
        df['value'] = scaler.fit_transform(df[['value']])
        time_step = 30
        X, y = self.create_dataset(df[['value']], df.value, time_step)
        
    #   model = self.build_model(X)
        model = keras.models.load_model('.models/new_model.h5')

        X_pred = model.predict(X,batch_size=32)

        mae_loss = np.mean(np.abs(X_pred - X), axis=1)
        THRESHOLD = np.percentile(mae_loss, 92)
        
        label = (mae_loss > THRESHOLD ).astype(np.int32)
        zeros = np.zeros((time_step,1))
        
        label = np.concatenate((zeros,label)).astype(np.int32)
        df['label2'] = label
        return df
    
anomaly_detector = TimeSeriesAnomalyDetector()

f1_scores = []
if __name__ == '__main__':
    
    for filename in os.listdir(input_path):
        input_df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(input_df))
        result = anomaly_detector(input_df)
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')
        f1_scores.append(f1_report(result))
    
    print(np.mean(f1_scores))
