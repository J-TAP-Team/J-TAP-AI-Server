from ai.transfer import wct_main
from flask import Flask, request, send_file

app = Flask(__name__)


@app.route('/',methods=['POST'])
def transfer():
    content = request.files["content"]
    style = request.files["style"]
    # wct_main(content,style)

    return send_file(content,mimetype='image/jpg',attachment_filename='downloaded_file_name.png',as_attachment=False)


if __name__ == '__main__':
    app.run(port=8000,debug=True)