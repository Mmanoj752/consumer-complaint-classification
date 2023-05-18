from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'GET':
        complaint = request.GET.get('consumer_complaint_narrative', '')

        # Load the saved model object and TfidfVectorizer
        model_and_vectorizer = joblib.load('customer_classification_model_lr.pkl')
        print(model_and_vectorizer)
        log_reg_model, tfidf_vect = model_and_vectorizer

        # Transform the complaint text using the loaded vectorizer
        x_valid_tfidf = tfidf_vect.transform([complaint])

        # Make the prediction using the transformed input
        predictions = log_reg_model.predict(x_valid_tfidf)

        return render(request, 'result.html', {'predictions': predictions})
    else:
        return HttpResponse('Invalid request method.')
