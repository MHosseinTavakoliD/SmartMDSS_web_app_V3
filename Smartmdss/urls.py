from django.contrib import admin
from django.urls import include, path
from django.contrib.auth import views as auth_views
from web_smartmdss import views as web_smartmdss_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('web_smartmdss.urls'), name='Michigan_home'),

    path('login/', auth_views.LoginView.as_view(template_name='web_smartmdss/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('register/', web_smartmdss_views.register_user, name='register'),
    path('get-weather-data/<str:point_id>/', web_smartmdss_views.get_weather_data, name='get-weather-data'),
    path('demo-data/', web_smartmdss_views.demo_data, name='demo-data'),
    path('submit_suggestion/<str:point_id>/', web_smartmdss_views.submit_suggestion, name='submit_suggestion'),
    path('mobileMichigan/', web_smartmdss_views.Mobile_Michigan_home, name='mobile_Michigan_home')
]
