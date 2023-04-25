from flask import Flask, render_template, request

app = Flask(__name__)
app.debug = True


@app.route('/', methods=['GET'])
def home():
    return render_template('./index.html')


# @app.route('/',methods= ['POST'])
# def predict():

if __name__ == "__main__":
    app.run(debug=True)
