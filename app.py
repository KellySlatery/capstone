# imports
import pickle
import numpy as np
import showmetunes
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
    # Get the musical recommendations given the user input
    response = user_input['UserInput']
    recs = showmetunes.recommend(response)
    return render_template('results.html', rec1=recs[0], rec2=recs[1], rec3=recs[2]) # Corresponds to template docs

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # python ./app_starter.py
    app.run(debug=True)
