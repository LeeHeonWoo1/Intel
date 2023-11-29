from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
import numpy as np
import pandas as pd
from keras.models import Sequential
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
from keras.utils import to_categorical

# CSV 파일 읽기
df = pd.read_csv(r'D:\Intel\NLP_Project\modeling\Merged_NewsResults copy2.csv')

# 입력 데이터와 타겟 데이터 생성
encoder_texts = df['본문'].astype(str).tolist()
decoder_texts = df['제목'].astype(str).tolist()

# 데이터 분리
encoder_texts_train, encoder_texts_test, decoder_texts_train, decoder_texts_test = train_test_split(
    encoder_texts, decoder_texts, test_size=0.2, random_state=42)

# KoNLPy 형태소 분석기 초기화
okt = Okt()

# KoNLPy를 사용하여 텍스트 토큰화
def tokenize(text):
    return okt.morphs(text)

# 토큰화된 텍스트로 대체
encoder_texts_train = [' '.join(tokenize(text)) for text in encoder_texts_train]
encoder_texts_test = [' '.join(tokenize(text)) for text in encoder_texts_test]

decoder_texts_train = [' '.join(tokenize(text)) for text in decoder_texts_train]
decoder_texts_test = [' '.join(tokenize(text)) for text in decoder_texts_test]

# 토큰화된 텍스트를 다시 시퀀스로 변환
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(encoder_texts_train)
num_encoder_tokens = len(tokenizer_encoder.word_index) + 1

tokenizer_decoder = Tokenizer()
tokenizer_decoder.fit_on_texts(decoder_texts_train)
num_decoder_tokens = len(tokenizer_decoder.word_index) + 1

encoder_sequences_train = tokenizer_encoder.texts_to_sequences(encoder_texts_train)
encoder_sequences_test = tokenizer_encoder.texts_to_sequences(encoder_texts_test)

decoder_sequences_train = tokenizer_decoder.texts_to_sequences(decoder_texts_train)
decoder_sequences_test = tokenizer_decoder.texts_to_sequences(decoder_texts_test)

# 시퀀스를 패딩하여 모든 시퀀스 길이를 맞춤
max_encoder_seq_length = max(len(seq) for seq in encoder_sequences_train)
max_decoder_seq_length = max(len(seq) for seq in decoder_sequences_train)

encoder_input_data_train = pad_sequences(encoder_sequences_train, maxlen=max_encoder_seq_length, padding='post')
encoder_input_data_test = pad_sequences(encoder_sequences_test, maxlen=max_encoder_seq_length, padding='post')

decoder_input_data_train = pad_sequences(decoder_sequences_train, maxlen=max_decoder_seq_length, padding='post')
decoder_input_data_test = pad_sequences(decoder_sequences_test, maxlen=max_decoder_seq_length, padding='post')

# BiLSTM 모델 만들기
latent_dim = 256

# 인코더 정의
encoder_inputs = Input(shape=(max_encoder_seq_length,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 디코더 정의
decoder_inputs = Input(shape=(max_decoder_seq_length,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

##################################################################################################################
# Load the saved model
model = load_model(r'D:\Intel\NLP_Project\modeling\newstitleModel.h5')

# Load the tokenizers
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(encoder_texts_train)
tokenizer_decoder = Tokenizer()
tokenizer_decoder.fit_on_texts(decoder_texts_train)

# Define the maximum sequence lengths
max_encoder_seq_length = max(len(seq) for seq in encoder_sequences_train)
max_decoder_seq_length = max(len(seq) for seq in decoder_sequences_train)

# KoNLPy 형태소 분석기 초기화
okt = Okt()

# Function to tokenize input text
def tokenize_input(text):
    return ' '.join(okt.morphs(text))

# Create a model for inference (without training) for the encoder
# Create a model for inference (without training) for the encoder
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Create a separate decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Function to generate title based on user input
def generate_title(input_text):
    # Tokenize the input text
    input_text = tokenize_input(input_text)
    # Convert the input text to sequence
    input_seq = tokenizer_encoder.texts_to_sequences([input_text])
    # Pad the input sequence
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    # Get the output from the encoder model
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)

    # Initialize target sequence with START_TOKEN
    start_token = tokenizer_decoder.word_index.get('START_TOKEN', None)
    if start_token is None:
        start_token = 1  # Fallback to the index of the first word in the vocabulary

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    # Generate titles word by word
    stop_condition = False
    decoded_title = ''
    while not stop_condition:
        # Get the output from the decoder model
        output_tokens, state_h, state_c = decoder_model.predict([target_seq] + [state_h, state_c])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer_decoder.index_word.get(sampled_token_index, '')
        decoded_title += ' ' + sampled_char
        # Exit condition: either hit max length or find the stop token
        if (sampled_char == 'END_TOKEN' or len(decoded_title) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_title

# Get user input
user_input = input("Enter the input text: ")
# Generate and print the title
generated_title = generate_title(user_input)
print("Generated Title:", generated_title)