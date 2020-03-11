# imports
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify, Response

# initialize the flask app
app = Flask('showMeTunes')

# route 1: hello world # Using a decorator (@)
@app.route('/')
# return a simple string
def home():
    return 'All the musical recommendations you could ever want (or more)-- coming soon! :--)'

# route 4: show a form to the user
@app.route('/form')
# use flask's render_template function to display an html page
def form():
    # We want to serve html as a file
    return render_template('form.html')

# route 5: accept the form submission and do something fancy with it
@app.route('/submit') # MUST BE /submit to match action attribute on the form!
# load in the form data from the incoming request
def form_submit():
    user_input = request.args

    # Have to set up features in the SAME ORDER the model expects them
    response = user_input['UserInput']

    # somehow load in a .py that will run and calculate!!!

    # FROM CLASS, NOT FOR ME TO USE -- FIGURE OUT HOW TO RUN MY FUNCTION
    model = pickle.load(open('assets/model.p', 'rb'))
    # filepath, 'rb'=read in binary format
    preds = model.predict(X_test)
    pred = round(preds[0], 2)

    # jsonify({'prediction': pred})

    return render_template('results.html', rec1=recs[0], rec2=recs[1], rec3=recs[2]) # Corresponds to template docs

# manipulate data into a format that we pass to our model

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # python ./app_starter.py
    app.run(debug=True)
