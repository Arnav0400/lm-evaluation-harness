import json
f = open("llama2-glora4-sharegpt-BS=32/mmlu-5shot-ckpt=5.json")
data = json.load(f)
acc = 0
for key in data['results']:
    acc += data['results'][key]['acc_norm']

print(acc/len(data['results']))