from flask import Flask,render_template
from flask import request
import pickle
import numpy as np 

filename='diabetes'
classifier=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('risk_score.html')

@app.route('/check-diabetes')
def home1():
    return render_template('diabetes.html')

@app.route('/predict-form-page')
def back():
    return render_template('diabetes.html')

@app.route('/risk_value', methods=['GET','POST'])
def risk_value():
        Age=int(request.form['age'])
        Gender=request.form['gender']
        Waist=int(request.form['waist'])
        family_history=request.form['db']
        diet=request.form['healthdiet']

        risk_score=0

        if Age:
            if Age<35:
                risk_score+=0
            elif Age>=35 and Age <=49:
                risk_score+=20
            else:
                risk_score+=30

        if Waist:
            if Waist<80 and Gender=='female':
                risk_score+=0
            elif Waist>=80 and Waist<=89 and Gender=='female':
                risk_score+=10
            elif Waist>=90 and Gender=='female':
                risk_score+=20    
            elif Waist<90 and Gender=='male':
                risk_score+=0
            elif Waist>=90 and Waist<=99 and Gender=='male':
                risk_score+=10
            elif Waist>=100 and Gender=='male':
                risk_score+=20
        if family_history:
            if family_history=='1':
                risk_score+=0
            elif family_history=='2':
                risk_score+=10
            else:
                risk_score+=20
        if diet:
            if diet=='1':
                risk_score+=0
            elif diet=='2':
                risk_score+=10
            elif diet=='3':
                risk_score+=20
            else:
                risk_score+=30
        if risk_score<40:
            risk='LOW'
        elif risk_score>=40 and risk_score<=70:
            risk='MODERATE'
        else:
            risk='HIGH' 
        return render_template('risk_score.html',risk=risk,risk_score=risk_score)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        preg=int(request.form['Pregnancies'])
        glucose=int(request.form['Glucose'])
        bp=int(request.form['BP'])
        st=int(request.form['SK'])
        insulin=int(request.form['Insulin'])
        bmi=float(request.form['BMI'])
        dpf=float(request.form['DPF'])
        age=int(request.form['Age'])

        data=np.array([(preg,glucose,bp,st,insulin,bmi,dpf,age)])
        my_prediction=classifier.predict(data)

        return render_template('result.html',prediction=my_prediction)
if __name__=='__main__':
    app.run()