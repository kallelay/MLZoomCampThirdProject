
import requests
import json
import pickle

with open("df_test.bin", 'rb') as f: df_test = pickle.load(f)
with open("y_test.bin", 'rb') as f: y_test = pickle.load(f)


while True:
    id = input("Give me an ID from the dataset:")
    try:
        id = int(id)
        example = df_test.iloc[id].to_json()
        expected_res = y_test[id]

        print(example)

        url = "http://127.0.0.1:9696/predict"
        results = requests.post(url, json=example).json()
        print(df_test.iloc[id:id+1,:])
        print("error_type of %d : %s" % (id, results["fail_type"]))

        print("(Real Database results: %s )" % expected_res)
    except Exception as e:
        print(e)