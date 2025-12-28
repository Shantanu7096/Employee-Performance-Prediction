from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open('model.pkl', 'rb'))

# Mappings
q_map = {'Quarter1':0, 'Quarter2':1, 'Quarter3':2, 'Quarter4':3, 'Quarter5':4}
d_map = {'finishing':0, 'sweing':1}
day_map = {'Monday':0, 'Saturday':1, 'Sunday':2, 'Thursday':3, 'Tuesday':4, 'Wednesday':5}

@app.route('/')
def home(): return render_template('home.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/predict')
def predict(): return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        inputs = [
            q_map[request.form['quarter']],
            d_map[request.form['dept']],
            day_map[request.form['day']],
            int(request.form['team']),
            float(request.form['target']),
            float(request.form['smv']),
            float(request.form['wip']) if request.form['wip'] else 0,
            int(request.form['ot']),
            int(request.form['inc']),
            float(request.form['idle_t']),
            int(request.form['idle_m']),
            int(request.form['style']),
            float(request.form['workers']),
            int(request.form['month'])
        ]
        
        prediction = model.predict([np.array(inputs)])[0]
        score = round(float(prediction), 4)

        # Performance Status Logic
        if score >= 0.80:
            status, color = "Highly Productive", "#238636"
        elif score >= 0.65:
            status, color = "Average Productivity", "#d29922"
        else:
            status, color = "Low Productivity", "#f85149"

        return render_template('submit.html', score=score, category=status, category_color=color)
    except Exception as e:
        return f"Form Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)