from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="home"),
    path('render_prediction/', views.render_prediction, name="render_prediction"),
    path('prediction/', views.prediction, name="prediction"),
    path('prediction/', views.predict_with_content, name="predict_with_content"),
    path('title_generation/', views.title_generation, name="title_generation"),
    path('render_text_sum/', views.text_summarization, name="render_text_summarization"),
    path('result_text_sum/', views.text_sum, name="text_summarization"),
]