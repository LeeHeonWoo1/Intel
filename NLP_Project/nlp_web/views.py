from django.http import HttpResponse, request
from django.shortcuts import render
from keras.models import load_model
from konlpy.tag import Mecab
import pickle
from . import use_model
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

def index(request):
    return render(request, "nlp_web/main.html")

def render_prediction(request):
    return render(request, "nlp_web/prediction.html")

def prediction(request):
    if request.method == 'POST':
        new_sentence = request.POST.get('new_sentence')
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        loaded_model = load_model(r'D:\Intel\NLP_Project\modeling\model\with_168783_data.h5')
        mecab = Mecab(r"C:\mecab\mecab-ko-dic") 
        
        with open(r"D:\Intel\NLP_Project\modeling\model\tokenizer.pickle", "rb") as path:
            tokenizer = pickle.load(path)
        
        result = use_model.predict(new_sentence, mecab, tokenizer, stopwords, loaded_model)
        return render(request, "nlp_web/predict_result.html", {"result":result})
    
    return HttpResponse("잘못된 요청입니다.")

def predict_with_content(request):
    if request.method == "POST":
        new_title = request.POST.get("new_sentence")
        new_content = request.POST.get("new_content")
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        model = load_model(r"D:\Intel\NLP_Project\modeling\new_model\model\128_hidden_64batch_adam.h5")
        mecab = Mecab(r"C:\mecab\mecab-ko-dic")
        
        with open(r"D:\Intel\NLP_Project\modeling\new_model\tokenizer\1st_tokenizer_title.pickle", "rb") as path:
            tokenizer1 = pickle.load(path)
            
        with open(r"D:\Intel\NLP_Project\modeling\new_model\tokenizer\1st_tokenizer_content.pickle", "rb") as path:
            tokenizer2 = pickle.load(path)
            
        result = use_model.prediction_with_content(new_title, new_content, mecab, tokenizer1, tokenizer2, stopwords, model)
        return render(request, "nlp_web/predict_result.html", {"result":result})
    
def text_summarization(request):
    return render(request, "nlp_web/text_summarization.html")

def text_sum(request):
    if request.method == "POST":
        tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

        text = request.POST.get("text_content")
        text = text.replace('\n', ' ')

        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
        result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        print(result)
        return render(request, "nlp_web/result_text_sum.html", {"result":result})
    
def title_generation(request):
    user_keyword = ''
