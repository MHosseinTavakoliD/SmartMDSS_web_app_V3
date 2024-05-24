from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=False)
    phone = forms.CharField(required=False)
    user_type = forms.ChoiceField(choices=UserProfile.user_type.field.choices, required=False)
    city = forms.CharField(required=False)
    county = forms.CharField(required=False)
    state = forms.CharField(required=False)

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'email', 'phone', 'user_type', 'city', 'county', 'state']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        # Remove any creation or handling of UserProfile here, handle it externally if needed
        return user

from django import forms

class PointOfInterestForm(forms.Form):
    road_name = forms.CharField(label='Road Name', max_length=100, required=False)
    latitude = forms.FloatField(label='Latitude', initial=49.00)
    longitude = forms.FloatField(label='Longitude', initial=-85.00)
      # New field

    road_type = forms.ChoiceField(choices=[
        ('highway', 'Highway'),
        ('main_road', 'Main Road'),
        ('secondary_road', 'Secondary Road'),
        ('local', 'Local'),
        ('county', 'county'),
    ], required=False)
    winter_maintenance = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Typically There is Winter Operation Maintenance for this Road', required=False)
    operation_plowing = forms.BooleanField(label='Type of Operation: Plowing', required=False, initial=False)
    operation_salting = forms.BooleanField(label='Type of Operation: Salting', required=False, initial=False)
    salt_type = forms.CharField(label='If salt is available, typically what salt you use:', required=False)
    plowing_per_day = forms.IntegerField(label='How many plowing is possible for your agency to do on this point per day', min_value=0, required=False)
    salting_per_day = forms.IntegerField(label='How many salting is possible for your agency to do on this point per day', min_value=0, required=False)
    share_point = forms.BooleanField(        label='OK to share this point with other users?*',        required=False,        initial=False)

