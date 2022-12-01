from email.mime import application
import numpy as np
from flask import Flask, request,render_template
import pickle

application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Chance of success would be {}'.format(output))

if __name__ == "__main__":
    application.run(debug=True)