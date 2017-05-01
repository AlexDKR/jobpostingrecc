"""
Contains main routes for the Prediction App
"""
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
from . import app, estimator, tfidf_vectorizer, lsa_vectorizer
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

class PredictForm(Form):
    """Fields for Predict"""
    input = fields.TextAreaField('Job keywords:', validators=[Required()])
    submit = fields.SubmitField('Submit')


@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    prediction = None
    if form.validate_on_submit():
        submitted_data = form.data
        input = submitted_data['input'].encode('utf-8')
        lsa_skills = lsa_vectorizer.transform(tfidf_vectorizer.transform([input]))
        sim_scores = np.dot(lsa_skills, estimator.T)
        sim_jobs = np.argsort(-sim_scores)[0][:5]
        appclient = MongoClient(port=27017, host='54.201.68.29')
        db = appclient.jobdescriptiondb
        jobmongo = db.jobapp
        job_res = []
        text = ''
        for job in sim_jobs:
            if sim_scores[0][job] > 0:
                job_hit = jobmongo.find_one({'df_ind': job}, {'job_url': 1,'title': 1,'desc': 1})
                if not job_hit['title']:
                    title = job_hit['title']
                else:
                    title = job_hit['title'].strip()
                job_res.append((title, job_hit['job_url']))

        prediction = job_res
    return render_template('index.html', form=form, prediction=prediction)
