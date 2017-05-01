"""
Create Application Object
"""
from flask import Flask
import pickle
from sklearn.externals import joblib

from flask import Flask
import pickle
from sklearn.externals import joblib
app = Flask(__name__)
app.config.from_object('app.config')
estimator = joblib.load('models/lsa_mat.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_notitles_vectorizer.pkl')
lsa_vectorizer = joblib.load('models/lsa_vectorizer.pkl')
from .views import *

@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return (
     render_template('404.html'), 404)
# okay decompiling __init__.pyc

