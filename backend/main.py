from flask import Flask, render_template
import os

app = Flask(__name__,
            template_folder=('../frontend/templates'),
            static_folder=('../frontend/static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions')
def questions():
    qa = [
        {"question": "What social and health services will the elderly population require in the coming years, from 2025 to 2030?", "answer": "Answer 1"},
        {"question": "Are there differences according to the regions of Spain and urban/rural areas?", "answer": "Answer 2"},
        {"question": "And according to the profile of the elderly (archetypes)?", "answer": "Answer 3"},
        {"question": "What role will digital technology play?", "answer": "Answer 4"},
    ]
    return render_template('questions.html', qa=qa)

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

@app.errorhandler(500)
def internal_error(error):
    return "500 error: " + str(error), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error('Unhandled Exception: %s', (e))
    return "Unhandled exception: " + str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

