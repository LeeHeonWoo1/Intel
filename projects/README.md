### 8/22 ~ 9/15 주차별 프로젝트

<hr>

### 1주차 : 종이 헬리콥터 회귀 분석

#### 프로젝트 목표
- 체공 시간이 가장 긴 변수 조합을 찾는다
- 체공 시간을 정확하게 예측할 수 있는 모델을 만든다.

#### 1일차 진행 현황
15건의 데이터 생성, 변수의 경우 날개 길이, 날개 폭, 몸통 길이, 다리 길이의 4가지로 구성, 신경망 구축

<div align = center><img src="./image/dataframe of paper helicopter.png"></div>

```python
input_layer = Input(shape = (4, ))
x = Dense(64, activation = "relu")(input_layer)
x = Dense(64, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
output_layer = Dense(1)(x)

optim = Adam(learning_rate = 0.001)

model = Model(input_layer, output_layer)
model.compile(optimizer = optim, loss = "mean_squared_error")
```
별도의 정규화 과정이 없기 때문에 출력층의 활성화 함수는 항등 함수로 활성화

테스트 데이터가 없기에 훈련 데이터를 그대로 집어넣어 예측을 진행했다.

<div align = center><img src="./image/dataframe for predictions of paper helicopter.png"></div>

#### 2일차 목표
- scikit learn으로 모델링, 내부 프로세싱 명확하게 이해하기