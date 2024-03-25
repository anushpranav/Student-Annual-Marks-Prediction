from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
     
# Load the dataset and train the model
data = pd.read_csv("marks.csv")  # Replace "marks.csv" with the path to your dataset
X = data.drop(columns=['Name', 'ID', 'Final Mark'])  
y = data['Final Mark']  
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Function to get prediction and confidence interval
def get_prediction(quarterly_mark, halfyearly_mark):
    input_data = pd.DataFrame({
        'Quarterly Mark': [quarterly_mark],
        'Halfyearly Mark': [halfyearly_mark]
    })
    final_mark_prediction = model.predict(input_data)
    predictions = [model.predict(input_data)[0] for _ in range(1000)]
    confidence_interval = np.percentile(predictions, [2.5, 97.5])
    return final_mark_prediction[0], confidence_interval

@app.route("/", methods=["GET", "POST"])
def predict_final_mark():
    if request.method == "POST":
        quarterly_mark = float(request.form["quarterly_mark"])
        halfyearly_mark = float(request.form["halfyearly_mark"])
        final_mark_prediction, confidence_interval = get_prediction(quarterly_mark, halfyearly_mark)
        return render_template("index.html", prediction=final_mark_prediction, confidence_interval=confidence_interval)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
