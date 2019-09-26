import load_data

[tr_data, va_data, te_data] = load_data.load_data_wrapper()

print(list(tr_data)[0][1])

print(list(va_data)[0][1])

print(list(te_data)[0][1])
