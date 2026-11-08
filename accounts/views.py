from django.contrib.auth import login
from django.shortcuts import redirect, render

from assistant.views import NEW_TOKEN_SESSION_KEY

from .forms import RegistrationForm
from .services import register_organization


def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard:home')
    form = RegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = register_organization(
            form.cleaned_data['org_name'], form.cleaned_data['email'], form.cleaned_data['password']
        )
        login(request, user)
        request.session[NEW_TOKEN_SESSION_KEY] = user.new_assistant_token
        return redirect('dashboard:home')
    return render(request, 'accounts/register.html', {'form': form})
