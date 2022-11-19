# Libraries used
import json
from logging_error import get_logger
import requests
from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

write = get_logger(__name__)
resp = {}

''' Fetch data from request and insert in log file '''
class service(Resource):
    def post(self):
        # Get data from request
        req = request.get_json()
        
        try:
            # Success, write data in log file
            write.debug(f'{req["status"]}|{req["service_name"]}|{req["error_code"]}|{req["exception_type"]}|{req["file_name"]}|{req["line_number"]}|{req["error_info"]}')
            resp["log"] = "success,{}".format(200)
        except IOError as err:
            # Fail case
            resp["log"] = "fail,{}".format(500)
    
        return resp

api.add_resource(service,'/error_handler/')
if __name__ == '__main__':
    
    # Seeting App Environment
    app.config["ENV"] = "dev"
    # Running app on specified port
    app.run(
        threaded=True,
        host="127.0.0.1",
        port=5000
    )