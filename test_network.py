import load_data
import network
import pickle

tr_data, va_data, te_data = load_data.load_data_wrapper()
nw = network.Network([784, 100, 10])
nw.SGD(30, 10, list(tr_data), 2.8, list(te_data))

model_file = "./model/1.model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(nw.bias, f)
    pickle.dump(nw.weight, f)
    pickle.dump(nw.sizes, f)
    print("model saved successfully!")
