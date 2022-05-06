import flask
from flask import Flask
from flask.json import jsonify
from flask.templating import render_template
from flask import session,request
from flask_session import Session
from flask_mysqldb import MySQL
import processor
app = Flask(__name__)
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'medixa'
 
mysql = MySQL(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def start():
     return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/health_checkup')
def health_checkup():
    return render_template('health_checkup.html')


@app.route('/bmi_calculator')
def bmi_calculator():
    return render_template('bmi_calculator.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/subscribe')
def subscribe():
    return render_template('subscribe.html')

@app.route('/initialResponse',methods=["POST"])
def initialResponse():
    firstres=request.form['question1']
    session["cat"]=firstres
    secondres=""
    if(session["cat"]=="Disease Prediction"):
        secondres="Please Enter Your Symptoms"
        return jsonify({"response":firstres,"sres":secondres})
    if(session["cat"]=="Drugs Information"):
        secondres="Enter the name of the medicine"
        return jsonify({"response":firstres,"sres":secondres})
    if(session["cat"]=="home_rem"):
        secondres="Enter the question you want to ask"
        return jsonify({"response":"General questions","sres":secondres})
    return start()
    

@app.route('/mainchat',methods=["POST"])
def startchat():
    question=request.form['questions']
    if(session["cat"]=="Disease Prediction"):
        disease=processor.answer(question)
        return jsonify({"response":disease,"cate":session["cat"]})
    elif(session["cat"]=="Drugs Information"):
        cursor=mysql.connection.cursor()
        cursor.execute("SELECT * from drugs WHERE name=%s",(question,))
        info=cursor.fetchall()
        return jsonify({"response":info,"cate":session["cat"]})
    elif(session["cat"]=="home_rem"):
        answer1=processor.general_question(question)
        return jsonify({"response":answer1,"cate":session["cat"]})
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)