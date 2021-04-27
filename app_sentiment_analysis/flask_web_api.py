from flask import (Flask, render_template, abort,url_for,
                   jsonify, request, redirect)

app=Flask(__name__)

@app.route("/", methods=["GET","POST"])
def welcome():
    if request.method == "POST":
        review = request.form['review']
        return render_template("welcome.html")
    else:
        return render_template("welcome.html")


if __name__ =='__main__':
    app.run(debug=True)