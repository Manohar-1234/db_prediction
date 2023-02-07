from flask import Flask,render_template
from flask import request
import pickle
import numpy as np 

filename='diabetes'
classifier=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        preg=int(request.form['Pregnancies'])
        glucose=int(request.form['Glucose'])
        bp=int(request.form['BP'])
        st=int(request.form['SK'])
        insulin=int(request.form['Insulin'])
        dpf=float(request.form['DP'])
        bmi=float(request.form['BMI'])
        age=int(request.form['Age'])

        data=np.array([(preg,glucose,bp,st,insulin,dpf,bmi,age)])
        my_prediction=classifier.predict(data)

        return render_template('result.html',prediction=my_prediction)
if __name__=='__main__':
    app.run()
