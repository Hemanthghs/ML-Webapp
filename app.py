from flask import Flask, render_template, request
import joblib 

model = joblib.load("model.sav")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    s_length = float(request.form['sepal_length'])
    s_width = float(request.form['sepal_width'])
    p_length = float(request.form['petal_length'])
    p_width = float(request.form['petal_width'])
    
    prediction = model.predict([[s_length, s_width, p_length, p_width]])
    
    if prediction[0] == 0:
        output = "Iris Setosa"
    elif prediction[0] == 1:
        output = "Iris Versicolor"
    else:
        output = "Iris Verginica"
        
    return render_template("index.html", result = output)
    
    
if __name__ == '__main__':
    app.run()  