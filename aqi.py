import pickle
from flask import Flask,render_template,request

import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model_rf.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2013.csv')
    prediction=model.predict(df.iloc[:,:-1].values)
    prediction=prediction.tolist()

    return render_template('result.html',text=prediction)


if __name__ == '__main__':
    app.run(debug=True)
