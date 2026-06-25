from flask import *
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import numpy as np

#now we import joblib files
le_edu=joblib.load('le_edu.pkl')
model_accuracy=joblib.load('accuracy.pkl')
model=joblib.load('modell.pkl')

web=Flask(__name__)
@web.route('/')

def home():
    return render_template('Edu_purchase_index.html')
@web.route('/predict',methods=['POST'])

def predict():
    try:
        data=request.json
        features=data['features']

        edu_input=str(features[2]).lower().strip()
        edu_enc=le_edu.transform([edu_input])[0]
        
        f1=float(features[0]) #58
        f2=float(features[1]) #39
        f4=float(features[3]) #37
        f5=float(features[4]) #78163
        f6=float(features[5]) #688

        final_feature=np.array([[f1,f2,f4,f5,f6,edu_enc]])
        prediction=model.predict(final_feature)
        final_prediction=int(prediction[0])
        

        result_text='Purchased the Product' if final_prediction == 1 else 'Not eligible to purchased'
        accuracy_precent=round(float(model_accuracy)*100)
        return jsonify({
            'prediction':final_prediction,
            'accuracy':accuracy_precent
        })

    except Exception as e:
        return jsonify({'error':str(e)})    
if __name__ =='__main__':
    web.run(host='0.0.0.0',port=5000,debug=True)        
