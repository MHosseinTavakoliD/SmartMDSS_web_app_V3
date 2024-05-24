from django.urls import path
from .views import home
from . import views
from django.views.generic import TemplateView


urlpatterns = [
    path('', views.Michigan_home, name='Michigan_home'),
    path('us/', views.home, name='home'),
    path('login/', views.login_user, name='login'),
    path('register/', views.register_user, name='register'),
    path('area_of_interest/', views.area_of_interest, name='area_of_interest'),
    path('delete_point/<str:point_id>/', views.delete_point, name='delete_point'),
    path('get-weather-data/<str:point_id>/', views.get_weather_data, name='get-weather-data'),
    path('demo/', TemplateView.as_view(template_name='web_smartmdss/demo.html'), name='demo'),
    path('submit_suggestion/<str:point_id>/', views.submit_suggestion, name='submit_suggestion'),
    path('Mobile_Michigan_home/', views.Mobile_Michigan_home, name='Mobile_Michigan_home')

]
