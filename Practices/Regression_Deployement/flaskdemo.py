from flask import Flask

app = Flask(__name__)

@app.route("/")
def demo():
    return "hi"

@app.route("/hello")
def demos():
    return "hello"

if __name__ == '__main__':
    app.run(debug=True)
