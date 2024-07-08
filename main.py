from flask import Flask, render_template

app = Flask(__name__,
            template_folder='templates', 
            static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

@app.route('/graphs/macroeconomic')
def macroeconomic():
    return render_template('macroeconomic.html')

@app.route('/graphs/hospitalisation')
def hospitalisation():
    return render_template('hospitalisation.html')

@app.route('/graphs/other')
def other():
    return render_template('other.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

