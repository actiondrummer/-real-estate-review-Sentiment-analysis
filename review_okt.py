import os
import pandas as pd
from konlpy.tag import Okt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (cnt / len(nested_list)) * 100))

def csv_add():###리뷰 폴더만 csv 합치기
    #폴더안의 파일들 리스트형태로 변경
    file_list=os.listdir("./review/")
    score_list = []
    desc_list = []
    # 추출해야되는 데이터
    extraction_data_score = ['score', 'trafficScore', 'careScore', 'residentScore', 'aroundScore']
    extraction_data_desc = ['desc', 'trafficDesc', 'careDesc', 'residentDesc', 'aroundDesc']

    for file in file_list:
        test_data=pd.read_csv("./review/"+file)
        for i in range(len(test_data)):
            for j in extraction_data_score:
                score=test_data[j].iloc[i]
                score_list.append(score)
            for j in extraction_data_desc:
                txt = test_data[j].iloc[i]
                desc_list.append(txt)
    data={"desc":desc_list,"score":score_list}
    test=pd.DataFrame(data,columns=["desc","score"])
    test.to_csv("output.csv")

def classification(): ###합친 리뷰데이터의 긍정 부정 분류
    df=pd.read_csv("output.csv")
    del df["Unnamed: 0"]
    #score 3인거 삭제
    idx_3=df[df['score']==3].index
    df=df.drop(idx_3)
    #score 4,5 ->1 아니면 0
    df=df.assign(outcome=(df['score']>2).astype(int))
    df.to_csv("output_result.csv")
    refine()

def refine():
    test_data = pd.read_csv("output_result.csv")
    test_data.drop_duplicates(subset=['desc'], inplace=True)  # desc 열에서 중복인 내용이 있다면 중복 제거
    test_data['desc'] = test_data['desc'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
    test_data['desc'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any')  # Null 값 제거
    print('전처리 후 테스트용 샘플의 개수 :', len(test_data))
    #test_data.to_csv("train_result.csv")
    word_okt(test_data)

def word_okt(train_data):### desc 형태소 분석
    stopword=[]
    okt = Okt()
    with open("stopword.txt","r") as f:
        lines=f.readlines()
        for i in lines:
            #print(i[:-1])
            stopword.append(i[:-1])
    X_train = []
    for sentence in train_data['desc']:
        temp=[]
        temp_X = okt.pos(sentence)  # 토큰화
        for i in range(len(temp_X)):
            if temp_X[i][1] in ['Adjective','Verb']: #형용사 동사 추출
                #print(temp_X[idx])
                temp.append(temp_X[i][0])
        #temp = [word for word in temp if not word in stopword]  # 불용어 제거
        if temp:
            X_train.append(temp)
    #print(X_train)
    model_gernation(X_train,train_data)

def model_gernation(X_train,train_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    threshold = 3
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    vocab_size = total_cnt - rare_cnt + 2
    print('단어 집합의 크기 :', vocab_size)
    tokenizer = Tokenizer(vocab_size, oov_token='OOV')
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    y_train = np.array(train_data['outcome'])
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    # 빈 샘플들을 제거
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    print(len(X_train))
    print(len(y_train))
    print('리뷰의 최대 길이 :', max(len(l) for l in X_train))
    print('리뷰의 평균 길이 :', sum(map(len, X_train)) / len(X_train))
    max_len = 30
    below_threshold_len(max_len, X_train)
    X_train = pad_sequences(X_train, maxlen=max_len)
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
    print(history)

if __name__=="__main__":
    csv_add()
    classification()
    refine()

