from flask import Flask,render_template,request
import random

app=Flask(__name__)
app.static_folder='static'

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/get")
def get_bot_response():
    userText=request.args.get('msg')
    list=["how are you?","I am fine","You seem good", "What's your name","Cool man", "Do u know who i am?"]
    return random.choice(list)
    
app.run(debug=True)