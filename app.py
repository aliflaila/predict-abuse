from flask import Flask, jsonify, request
import joblib
from bs4 import BeautifulSoup
import re
import flask
# https://www.tutorialspoint.com/flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = []
    clf = joblib.load('finalized_model.pickle')
    count_vect = joblib.load('vectorizer.pickle')
    text = request.form.get('review_text')
    #print(text)
    pred = clf.predict(count_vect.transform([text]))
    x = pred.toarray()
    print(x)
    if x[0,0]:
        prediction.append('Commenting') 
    if x[0,1]:
        prediction.append('Ogling')
    if x[0,2]:
        prediction.append('Groping')
    return flask.render_template('app.html', prediction_text='Predicted categories of abuse are {}'.format(prediction))
    #return jsonify(prediction)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)