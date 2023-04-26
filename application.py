from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import os.path
from wtforms.validators import InputRequired
from src.pipeline.training_pipeline import TrainingPipeline

application = Flask(__name__)
app = application
app.config["SECRET_KEY"] = 'secretkey'
app.config["UPLOAD_FOLDER"] = 'notebooks\data'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/training", methods=['GET', 'POST'])
def training_model():
    if request.method == 'GET':
        form = UploadFileForm()
        # if form.validate_on_submit():
        #     # getting the file from webpage
        #     file = form.file.data  
        #     file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        #     return "File Uploaded successfully"
        return render_template("training.html", form=form)
    else:
        form = UploadFileForm() 
        if form.validate_on_submit():
            path:str = os.path.join("notebooks\data", "upload.csv")
            if os.path.isfile(path):   # Checking if that same file is in database and if present removing it
                os.remove(path)
            # getting the file from webpage
            file = form.file.data  
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            obj = TrainingPipeline()
            best_model, best_score, flag= obj.model_train()
            best_score = round(best_score*100, 2)
            if flag:
                st = f"Model trained on given dataset has chosen {best_model} model and having accuracy of {best_score}%"
            else:
                st = f"Given dataset is not appropriate, but model trained on inner dataset and choose {best_model} as best model with accuracy of {best_score}%"
            return render_template("trainingResult.html", final_model = best_model, final_score =best_score, final_st = st)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        final_new_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_df)

        results = round(pred[0],2)

        return render_template('results.html', final_result=results)







if __name__ == "__main__":
    app.run(host="0.0.0.0")    