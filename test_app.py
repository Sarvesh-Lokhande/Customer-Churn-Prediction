from flask import Flask
app = Flask(__name__)

@app.route("/")
def ok():
    return "Flask is working ✅"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=6060, debug=True)
