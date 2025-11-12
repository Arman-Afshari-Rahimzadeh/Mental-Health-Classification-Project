from flask import Flask, request, render_template
import pandas as pd

# Fix the moddule name to match your file
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET, POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    # POST handling
    try:
        # Collect inputers(names must match your html form input 'name' attributes)
        data = CustomData(
            birth_year=float(request.form.get('birth_year')),
            potential_therapy=float(request.form.get('potential_therapy')),
            employment_status=request.form.get('employment_status')
        )

        # Convert to Dataframe for the pipeline
        pred_df = data.get_data_as_frame()
        print("Input dataframe:\n", pred_df())

        # Create pipeline and call predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Pass result to template
        return render_template('home.html', results=results[0])
    
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('home.html', error=str(e))
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")