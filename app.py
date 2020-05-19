# Adapted from Riley's GA lesson: 12.02-lesson-flask-basics

# Imports
import numpy as np
import showmetunes
from flask import Flask, request, render_template, jsonify, Response

# Initialize the flask app
app = Flask('showMeTunes')

# Initialize the first route (home page)
@app.route('/')

# Define what happens on that page: return a string
def home():
    return 'All the musical recommendations you could ever want (or more)-- coming soon! :--)'

# Initialize the input page: from an html file that we'll display
@app.route('/form')

# Define what happens: show the html input form
def form():
    return render_template('form.html')

# Initialize the results page: what happens when the user inputs data
@app.route('/submit')

# Define what happens: load form data and return recommendations
def form_submit():
    user_input = request.args
    # Get the musical recommendations given the user input
    response = str(user_input['UserInput'])
    recs = showmetunes.recommend(response)
    # Track the output
    showmetunes.track_output(response, recs)
    # Show html output on page
    return render_template('results.html', rec1=recs[0], rec2=recs[1], rec3=recs[2]) # Corresponds to template docs

# Run when python script is called (debug=True)
if __name__ == '__main__':
    app.run(debug=True)
