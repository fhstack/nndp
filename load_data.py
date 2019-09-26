import  pickle
import  numpy as np
import  gzip

def load_data_wrapper():
    tr_data, va_data, te_data = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
    training_results = [vectorized_result(y) for y in tr_data[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_data[0]]
    validation_data = zip(validation_inputs, va_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_data[0]]
    test_data = zip(test_inputs, te_data[1])
    return (training_data, validation_data, test_data)

def load_data():
    f =  gzip.open("./data/mnist.pkl.gz", "rb")
    traning_data, validation_data, test_data = pickle._load(f, encoding = "latin1")
    f.close()
    return (traning_data, validation_data, test_data)

def vectorized_result(j):
    v = np.zeros((10,1))
    v[j] = 1.0
    return v
