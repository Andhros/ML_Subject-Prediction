import requests
import json
# headers = {"accept": "application/json",
#            "Content-Type": "application/json"}
to_predict = {"age": 21, "Medu": 1, "Fedu": 1, "traveltime": 1, "studytime": 1, "failures": 3, "famrel": 5, "freetime": 5, "goout": 3, "Dalc": 3, "Walc": 3, "health": 3, "absences": 3, "G1": 10, "G2": 8}
to_predict = json.dumps(to_predict)
url = 'https://127.0.0.1:8000/predict/'
r = requests.post(url, data=to_predict)
print(r.text)
