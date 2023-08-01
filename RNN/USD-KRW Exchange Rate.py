import numpy as np
import matplotlib.pyplot as plt, matplotlib as mlp
import math
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

mlp.rcParams["font.family"] = "Malgun Gothic"

df = pandas.read_csv('./RNN/DEXKOUS.csv', usecols=[1], thousands=",", skipfooter = 3, engine = "python").sort_index(ascending=False) # 종가 데이터만 가지고와서 작업
dataset = df.values # ndarray 타입으로 변환
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1)) # 정규화 객체 생성
dataset = scaler.fit_transform(dataset)     # 정규화 진행

# train, test를 9:1 비율로 분할
train_size = int(len(dataset) * 0.9)         
test_size = len(dataset) - train_size      
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# 데이터 셋 구성
def create_dataset(dataset, maxlen):         # maxlen은 다음 시간 영역 예측을 위한 앞쪽 시간대의 스텝 수
    dataX, dataY = [], []                    # maxlen이 3이라고 가정하면, x는 t-2, t-1, t시점의 종가이며, y는 t+1시점의 종가로 구성된다.
    for i in range(len(dataset)-maxlen-1):   
        a = dataset[i:(i+maxlen), 0]         # t-2, t-1, t시점의 종가 추출
        dataX.append(a)                      # X데이터로 구성
        dataY.append(dataset[i + maxlen, 0]) # t+1 시점의 종가 추출, Y데이터로 구성
    return np.array(dataX), np.array(dataY)  

maxlen = 3
trainX, trainY = create_dataset(train, maxlen)
testX, testY = create_dataset(test, maxlen)
 
print (trainX.shape)
print (trainY.shape)
 
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) # 모든 데이터의 개수, 일수(하루 단위), 스텝 수(3일 전의 데이터부터 분석)로 학습 데이터를 구성
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))     

print(trainX.shape)
print(testX.shape)

# 모델링
model = Sequential()
model.add(LSTM(64, input_shape=(1, maxlen)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 학습
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=1) # 지난 3일간의 종가들과 다음날의 종가를 기반으로 학습을 진행

# 예측
trainPredict = model.predict(trainX) # 학습 데이터에 대한 예측
testPredict = model.predict(testX)   # 테스트 데이터에 대한 예측

# 예측된 결과를 원래의 형태로 반환(정규화했던 형태에서 비정규화 형태로 다시 변환)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# RMSE 지표 측정
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
 
trainPredictPlot = np.empty_like(dataset)                           # 시각화를 위해 dataset의 길이만큼 배열 생성
trainPredictPlot[:, :] = np.nan                                     # nan값으로 초기화를 진행하고
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] = trainPredict # 스텝(3)까지는 nan값으로 유지하고, 나머지는 예측된 결과로 채운다.

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset), color ="g", label = "row") # 원본 데이터를 초록색으로 시각화
plt.plot(trainPredictPlot,color="b", label="trainpredict")             # 훈련 데이터 예측 결과를 파란색으로 시각화
plt.plot(testPredictPlot,color="r", label="testpredict")               # 테스트 데이터 예측 결과를 빨간색으로 시각화
plt.title('환율 종가 예측결과') 
plt.legend()
plt.show()