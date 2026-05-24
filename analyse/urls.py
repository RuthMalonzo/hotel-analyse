from django.urls import path
from . import views

urlpatterns = [
    path('', views.accueil, name='accueil'),
    path('statistiques/', views.statistiques, name='statistiques'),
    path('visualisations/', views.visualisations, name='visualisations'),
    path('sentiments/', views.sentiments, name='sentiments'),
    path('interactive/', views.interactive, name='interactive'),
]