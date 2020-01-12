# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:23:41 2020

@author: nww73
"""
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def tokenize_sentence(data):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_data = []
    max_length = 0
    
    for sentence in data:
        tokens = tokenizer.tokenize(sentence)
        tokenized_data.append(tokens)
        if max_length<len(tokens): 
            max_length=len(tokens)
    
    return tokenized_data, max_length

#list_2d는 2차원 list이다. 2000개의 글과 각 글이 tokenized 된 단어들이다.
def word2num(list_2d):
    w2n_dic = dict()  # word가 key이고 index가 value인 dict
    n2w_dic = dict()  # index가 key이고 word가 value인 dict. 나중에 번호에서 단어로 쉽게 바꾸기 위해.
    idx = 1
    num_list = [[] for _ in range(len(list_2d))]   # 숫자에 매핑된 글의 리스트
    for k,i in enumerate(list_2d):
        if not i:
            continue
        elif isinstance(i, str): 
             # 내용이 단어 하나로 이루어진 경우, for loop으로 ['단어']가 '단'과 '어'로 나뉘지 않게 한다.
            if w2n_dic.get(i) is None:
                w2n_dic[i] = idx
                n2w_dic[idx] = i
                idx += 1
            num_list[k] = [dic[i]]
        else:
            for j in i:
                if w2n_dic.get(j) is None:
                    w2n_dic[j] = idx
                    n2w_dic[idx] = j
                    idx += 1
                num_list[k].append(w2n_dic[j])
    return num_list, w2n_dic, n2w_dic

def Evaluate(predicted, actual, labels):
    output_labels = []
    output = []

    # Calculate and display confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (False, True)\n- y-axis is predicted labels')

    # 정확도 : 정확하게 측정된 비율
    # Calculate precision, recall, and F1 score
    # np.trace(cm) : confusion_matrix의 대각 원소의 합 => 정확히 예측된 경우
    # np.sum(cm) : confusion_matrix의 모든 원소의 합
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    # 정밀도 : 정상이라고 예측한 것들 중 실제 정상인 값의 비율
    precision = precision_score(actual, predicted, average=None, labels=labels)
    # 재현율 : 실제 정상인 것들 중 정상이라고 예측한 비율
    recall = recall_score(actual, predicted, average=None, labels=labels)
    # F점수 : 정밀도와 재현율의 가중조화평균 가중치(베타)가 1인 경우 F1점수 라고 함
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels

    return output_df

df = pd.read_csv('data/BuildingTag_Napa.csv', encoding='utf-8')
tokenized_data = df['Description'].values.tolist()
tokenized_data, max_length = tokenize_sentence(tokenized_data)

y = df['Placard'].values.tolist()
X, w2n_dic, n2w_dic = word2num(tokenized_data)

hidden_layers = 70  # if you want to increase hidden layer of LSTM, change this.
epochs = 20
words_num = len(w2n_dic)
batch_size = 100

epochs_acc = []
epoch = range(10,110,10)
for epochs in epoch:

    # train:valid:test = 6:2:2
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=0.25, stratify=Y_train, random_state=42)

    X_train = sequence.pad_sequences(X_train, maxlen=max_length) # 50개 넘으면 자르고 안되면 0으로 채움
    X_validation = sequence.pad_sequences(X_validation, maxlen=max_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)

    # one_hot encoding
    # 2 -> [0,0,1]
    Y_train = np_utils.to_categorical(Y_train)
    Y_validation = np_utils.to_categorical(Y_validation)
    Y_test = np_utils.to_categorical(Y_test)


    # For Checkpoint
    # MODEL_SAVE_FOLDER_PATH = 'model/'
    # if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    #   os.mkdir(MODEL_SAVE_FOLDER_PATH)
    #
    # model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    # cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
    #                                 verbose=1, save_best_only=True)

    # Model
    model = Sequential()
    model.add(Embedding(words_num+1, len(X_train[0])))  # 사용된 단어 수 & input 하나 당 size
    model.add(LSTM(hidden_layers))
    model.add(Dense(len(Y_train[0]), activation='softmax'))  # 카테고리 수

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_validation,Y_validation))#, callbacks=[cb_checkpoint])
    model.save_weights('model/model.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    #
    # plt.plot(epochs, acc, 'b', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.yticks(np.arange(0.4, 1.0, 0.05))
    # plt.xticks(epochs)
    # plt.ylim(0.4, 1.0)
    # plt.title('Accuracy')
    # plt.legend()
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.xticks(epochs)
    # plt.title('Loss')
    # plt.legend()
    # plt.show()

    target_names = ['Green', 'Yellow', 'Red']
    Y_test_pred = model.predict(X_test)

    Y_test_class = np.argmax(Y_test, axis=1)
    Y_test_pred_class = np.argmax(Y_test_pred, axis=1)

    evaluation_result = Evaluate(actual=Y_test_class,
                                 predicted=Y_test_pred_class,
                                 labels=[0,1,2])

    pd.set_option('display.max_columns', 100)
    print(evaluation_result)  # show full results for first split only

    epochs_acc.append(val_acc[-1])

plt.plot(epoch, epochs_acc)
plt.title('Accuracy: Num of Layers')
plt.legend()
plt.tight_layout()
plt.show()