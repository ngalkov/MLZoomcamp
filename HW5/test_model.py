import pickle

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)
with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
score = predict_single(client, dv, model)
print(score)
