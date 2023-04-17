
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
#import processing as pr
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"]=True

@app.route("/",methods=["GET","POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        a = None
        b = None
        p = None
        try:
            a = pd.eval(request.form["a"])
        except:
            errors += "<p>{!r} is not a integer.</p>\n".format(request.form["a"])
        try:
            b = pd.eval(request.form["b"])
        except:
            errors += "<p>{!r} is not a integer.</p>\n".format(request.form["b"])
        try:
            p = pd.eval(request.form["p"])
        except:
            errors += "<p>{!r} is not a integer.</p>\n".format(request.form["p"])

        if a is not None and b is not None and p is not None:
            return '''
                <html>
                    <body>
                        <p>Below is the model with a={a}, b={b} (so n={n}) and p={p}:</p>
                        <p><a href="/">Click here to show an other model</a>
                    </body>
                </html>
            '''.format(a=a,b=b,n=a+b,p=p)
    return '''
        <html>
            <body>
                {errors}
                <p>Enter the coefficients of the Belyi drawing:<br>
                <small>(Here the p-valuation of the coefficient a must be greater or equal to the p-valuation
                of the coefficient b.)</small></p>
                <form method="post" action=".">
                    <p>Coefficient a: <input name="a" value="2*5**2"/> </p>
                    <p>Coefficient b: <input name="b" value="3*5"/> </p>
                    <p>Modulo p: <input name="p" value="5"/> </p>
                    <p><input type="submit" value="Show model" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)


app.run(host='0.0.0.0', debug=False)
