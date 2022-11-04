from flask import Flask, request
from  flask_restful import Resource, Api
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

model = pickle.load( open( "rf_model1.pkl", "rb" ) )

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        
        # getting predictions from the model
        res = model.predict_proba(df)
        res = res.tolist()
        max_value = max(res[0])
        max_index = res[0].index(max_value)
        if max_index == 1:
            res = f'There is a {round(max_value*100,2)}% probability this loan application will be approved.'
        else:
            res = f'There is a {round(max_value*100,2)}% probability this loan application will be rejected.'
        
        return res

# assign endpoint
api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)