from flask import Flask, render_template

app = Flask(__name__)


@app.route("/)
def index():
    return render_template("index.html")

@app.route("/more_love")
def more_love():
    things_i_love = [
        "Star Wars",
        "Coffee",
        "Cookies",
    ]
    return render_template("more_love.html", things_i_love=things_i_love)


if __name__ == "__main__":
    app.run(debug=True)