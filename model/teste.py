import requests
import json

to_predict = {
"age": 21,
"Medu": 1,
"Fedu": 1,
"traveltime": 1,
"studytime": 1,
"failures": 3,
"famrel": 5,
"freetime": 5,
"goout": 3,
"Dalc": 3,
"Walc": 3,
"health": 3,
"abscences": 3,
"G1": 10,
"G2": 8
}

to_predict = json.dumps(to_predict)

print(type(to_predict))

url = 'http://127.0.0.1:8000/predict/'
r = requests.post(url, data=to_predict)
r.json
