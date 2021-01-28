from django.urls import path
from . import views  # . means current directory


urlpatterns = [
    path('', views.home, name='app-home'),
    path('about/', views.about, name='app-about'),
    path('graph/', views.graph, name='app-graph'),
    path('search/', views.search, name='app-search'),
]
