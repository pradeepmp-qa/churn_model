from flask import Flask, render_template, request
import joblib

# initialse the application
app = Flask(__name__)

#load the model

from keras.models import load_model
model = load_model('20_aug_82.h5')

@app.route('/')
def churn():
    return render_template('form.html')


@app.route('/submit' , methods = ["POST"])
def form_data():

   CreditScore = request.form.get('CreditScore')
   Germany = request.form.get('Germany')
   Spain = request.form.get('Spain')
   Male = request.form.get('Male')
   Age = request.form.get('Age')
   Tenure = request.form.get('Tenure')
   Balance = request.form.get('Balance')
   NumOfProducts = request.form.get('NumOfProducts')
   HasCrCard = request.form.get('HasCrCard')
   IsActiveMember = request.form.get('IsActiveMember') 
   EstimatedSalary = request.form.get('EstimatedSalary')

   scaling = joblib.load('feature_sacling.pkl')

   data = scaling.transform([[CreditScore,Germany,Spain,Male,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])

   output = model.predict(data) > 0.5

   if output[0][0] == True:
        out = 'he\she will leave'
   else:
        out = 'he/she will stay'


   return render_template('predict.html' , data = f'Person {out}')

if __name__ == '__main__':
    app.run(debug = True)

