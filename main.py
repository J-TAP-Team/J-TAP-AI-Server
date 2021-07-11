from ai.transfer import wct_main
from flask import Flask, request, send_file
from ai.util import s3upload
from datetime import datetime

app = Flask(__name__)


@app.route('/',methods=['POST'])
def transfer():
    content = request.files["content"]
    style = request.files["style"]
    wct_main(content,style)

    with open('output.jpg', 'rb') as data:
        url = s3upload(data, "test")

    return str(url)

if __name__ == '__main__':
    app.run(port=8000,debug=True)