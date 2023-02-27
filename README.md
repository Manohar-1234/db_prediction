
#Web application development using flask
Flask is a Python-based microweb platform that allows users to add application functionality as if they were built into the framework itself. Below fig shows the basic file structures of the developed application and this development process comprises of four different program modules as follows:

• model.pkl- This contains the machine learning model to predict diabetes. As SVM provided the highest accuracy of 78.125% with all the features, we will integrate this as predictive model in the model.pkl file.

• app.py- This package includes Flask APIs that receive Diabetes information through GUI or API calls, compute the predicted value using our model, and return it.

•Template- The HTML form (index.html) in this folder allows the user to enter diabetes information and shows the expected outcome.

•Static- This folder contains the css file which has the styling required for our HTML form.

DiabetesPrediction/
			|--------model.pkl
			|--------static
			|--------template/
				|-----diabetes.html(main)
				|-----result.html
				|-----risk_score.html
			|--------app.py
