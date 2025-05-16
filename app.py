from flask import Flask, render_template, request, redirect, url_for

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Route for the main page (selection page)
@app.route('/')
def index():
    return render_template('main.html')

# Route to handle the selection form
@app.route('/select', methods=['POST'])
def select_option():
    selection = request.form.get('selection')
    
    if selection == 'end_to_end':
        # Redirect to the End to End solution page
        return redirect("http://localhost:5001/end_to_end")  # Port for app 1
    
    elif selection == 'separated_features':
        # Redirect to the Separated Features page
        return redirect("http://localhost:5002")  # Port for app 2

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run main app on port 5000
