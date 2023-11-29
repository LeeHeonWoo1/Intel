"""
각 url pattern에 맞는 함수를 실행하는 파일입니다.
"""

import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(new_sentence, mecab, tokenizer, stopwords, model):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-z ]','', new_sentence)
    new_sentence = mecab.morphs(new_sentence) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 50) # 패딩
    score = float(model.predict(pad_new, verbose=0)) # 예측
    if(score > 0.5):
        result = "상위 {:.2f}%의 게시글입니다. 흥미로운데요? ".format((1 - score) * 100)
    else:
        result = "하위 {:.2f}%의 게시글입니다. 조금은 아쉬운 것 같아요 :(".format(score * 100)
    
    return result

def prediction_with_content(new_title, new_content, mecab, tokenizer1, tokenizer2, stopwords, model):
    new_title = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-z ]', '', new_title)
    new_content = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-z ]', '', new_content)
    
    new_title = mecab.morphs(new_title)
    new_content = mecab.morphs(new_content)
    
    new_title = [word for word in new_title if not word in stopwords]
    new_content = [word for word in new_content if not word in stopwords]
    
    encoded_title = tokenizer1.texts_to_sequences([new_title])
    encoded_content = tokenizer2.texts_to_sequences([new_content])
    
    padded_title = pad_sequences(encoded_title, maxlen = 30)
    padded_content = pad_sequences(encoded_content, maxlen = 1000)
    
    new_data = np.concatenate((padded_title, padded_content), axis = 1)
    print(new_data.shape)
    
    score = float(model.predict(new_data, verbose=0))
    if(score > 0.5):
        result = "상위 {:.2f}%의 게시글입니다. 흥미로운데요? ".format(score * 100)
    else:
        result = "상위 {:.2f}%의 게시글입니다. 조금은 아쉬운 것 같아요 :(".format(score * 100)
    
    return result
