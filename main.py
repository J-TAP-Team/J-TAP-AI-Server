import ai.transfer
from flask import Flask, request

app = Flask(__name__)


@app.route('/',methods=['POST'])
def transfer():
    content = request.files["content"]
    style = request.files["style"]

    return content.filename


if __name__ == '__main__':
    app.run(port=8000,debug=True)