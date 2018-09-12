# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:37:02 2018

@author: paprasad
"""

from  predict_app import predict

from flask import Flask,jsonify,request,abort
app = Flask(__name__)

@app.route("/")
def main():
    return "<h1>Welcome!</h1><p>This welcome page for Smart Machine</p>"



@app.route('/todo/api/v1.0/tasks', methods = ['GET'])
def get_tasks():
    return jsonify( { 'tasks': 'pawan' } )



@app.route('/todo/api/v1.0/tasks1', methods=['POST'])
def create_task():
    if not request.is_json or not 'ticket_id' in request.json or not 'ticket_date' in request.json or not 'ticket_desc' in request.json:
        abort(400)
                       
    ticket_id = request.json['ticket_id']
    ticket_date = request.json['ticket_date']
    ticket_desc = request.json['ticket_desc']
    return jsonify({'Predicted': predict(ticket_desc)})


if __name__ == "__main__":
    app.run(debug=True)