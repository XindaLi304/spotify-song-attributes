import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


origin =pd.read_csv('.\\data.csv')
#data preprocess
df=origin.copy().dropna()
df_label=df.pop('target')
artists= df['artist']

# transform data type
title=df['song_title']
title.to_numpy()
print(title)
song_len =[]
for i in range(2017):
    #print(len(title[i]))
    song_len.append(len(title[i]))
df['song_length']=song_len

#feature_num=['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode',
            # 'speechiness','tempo','time_signature','valence','song_length']
feature_num=['danceability', 'speechiness', 'instrumentalness', 'duration_ms', 'valence']

# total view
pd.set_option('display.max_columns', None)
print(df.head(10))
print(df.describe())


# check correlation
#  turn the category into num
show=origin.copy()
show['song_length']=song_len
show['artist'] = LabelEncoder().fit_transform(show['artist'])
cor_mat=show.corr()
print(cor_mat)
plt.figure(figsize=(10,10))
sns.heatmap(show.corr(), annot=True)
plt.show()

prepeocessor=make_column_transformer(
    (StandardScaler(),feature_num)


)
def group_split(X,y,group,train_size=0.8):
    splitter =GroupShuffleSplit(train_size=train_size)
    train,valid=next(splitter.split(X,y,groups=group))
    return (X.iloc[train],y.iloc[train],X.iloc[valid],y.iloc[valid])

X_train,y_train,X_valid,y_valid= group_split(df,df_label,artists)

X_train=prepeocessor.fit_transform(X_train)
X_valid=prepeocessor.transform(X_valid)



early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)


model=keras.Sequential(
[layers.Dense(5,activation='relu',input_shape=(5,)),
 layers.Dense(5,activation='relu'),
 layers.Dense(5,activation='relu'),



 layers.Dense(1,activation='sigmoid')
]



)
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

early_stopping = callbacks.EarlyStopping(patience=100,min_delta=0.001,restore_best_weights=True)
history=model.fit(X_train,y_train, validation_data=(X_valid,y_valid),epochs=500, callbacks=None)

his_df=pd.DataFrame(history.history)
his_df['loss'].plot()
his_df['accuracy'].plot()
plt.show()
