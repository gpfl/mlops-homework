import requests

ride_dict = {"year": 2021, "month": 4}

url = "http://127.0.0.1:9696/predict"
response = requests.post(url, json=ride_dict)
print(response.json())
