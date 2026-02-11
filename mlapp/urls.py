from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),  
    path('explore/', views.explore, name='explore'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),  # HTML Form
    path('api/predict/', views.predict_api, name='predict_api'),  # API POST
    path('stocks/', views.explore_stocks, name='stocks'),
    path('stocks/<str:symbol>/', views.stock_detail, name='stock_detail'),
    path('history/', views.history_view, name='history'),
    path('insights/', views.insights_view, name='insights'),
    path('crypto/', views.explore_crypto, name='explore_crypto'),
    path('crypto/<str:symbol>/', views.crypto_detail, name='crypto_detail'),
    path('news/', views.financial_news, name='news'),
    path('sip-calculator/', views.sip_calculator, name='sip'),
    path('recommend/', views.recommend_investment, name='investment'),
    path('mutualfunds/', views.mutual_funds_list, name='mutual_funds_list'),
]
