#!/usr/bin/python3

import network
import pickle
import load_data

if __name__ == "__main__":
    nw = network.Network()
    model = "model/1.model.pkl"
    f = open(model, "rb")
    nw.bias = pickle.load(f)
    nw.weight = pickle.load(f)
    nw.sizes = pickle.load(f)
    _, _, te_data = load_data.load_data_wrapper()
    td = list(te_data)    
    correct = nw.evaluate(td)
    print("res: %d / %d" % (correct, len(td)))
