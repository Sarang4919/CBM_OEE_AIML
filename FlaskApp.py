from flask import Flask, jsonify
import multiclass_app
import binary_app
import RUL_Cox

# GET method, Link : http://127.0.0.1:5000/
sample_input_data = {
    "Lifetime" : 3,
    "Temperature" : 40,
    "Voltage" : 223,
    "Current" : 1,
    "Humidity" : 50,
    "VibrationX" : 1,
    "VibrationY" : 1,
    "VibrationZ" : 1,
    "Machine_Name" : 123
}

print(multiclass_app.predict_multiclass(sample_input_data))
print(binary_app.predict_binary(sample_input_data))
print(RUL_Cox.predict_RUL(sample_input_data))

app = Flask(__name__)

@app.route('/')
def Index():

    #-------------------------------------------------------------#
    # Add code to query for the latest data from MongoDB
    #-------------------------------------------------------------#

    #-------------------------------------------------------------#
    # POST that data to the ML Flask Server
    # Get the results JSON from the response.
    #-------------------------------------------------------------#

    #-------------------------------------------------------------#
    # Put the data back to MongoDB
    #-------------------------------------------------------------#


    return f"{multiclass_app.predict_multiclass(sample_input_data)}, {binary_app.predict_binary(sample_input_data)}, {RUL_Cox.predict_RUL(sample_input_data)}"

if __name__ == "__main__":
    app.run(debug=True)