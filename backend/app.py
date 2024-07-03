from flask import Flask, render_template

app = Flask(__name__,
            template_folder='../frontend', 
            static_folder='../frontend')

# Landing page route
@app.route('/')
def index():
    return render_template('index.html')

# Page to display answers to questions
@app.route('/questions')
def questions():
    # Static questions and answers
    qa = [
        {"question": "Question 1", "answer": "Answer 1"},
        {"question": "Question 2", "answer": "Answer 2"},
        {"question": "Question 3", "answer": "Answer 3"},
        {"question": "Question 4", "answer": "Answer 4"},
    ]
    return render_template('questions.html', qa=qa)

# Page to show static graphs
@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
