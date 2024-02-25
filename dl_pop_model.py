import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import h5py
import matplotlib.pyplot as plt
def get_dataset():
    Niter = 100
    path = "./pv_firing_rate/" # "./datasets/CA1 Basket/"

    target_firing_rate = []

    X = np.empty( (Niter, 20000, 2), dtype=np.float32)
    Y = np.empty( (Niter, 20000, 1), dtype=np.float32)

    for idx in range(Niter):
        filepath = "{path}/{i}.hdf5".format(path=path, i=idx)
        with h5py.File(filepath, mode='r') as h5file:
            # gexc.append( h5file["gexc"][:].ravel() )
            # ginh.append( h5file["ginh"][:].ravel() )

            X[idx, :, 0] = h5file["gexc"][:].ravel() / 40.0
            X[idx, :, 1] = h5file["ginh"][:].ravel() / 50.0

            #target_firing_rate.append(h5file["firing_rate"][:].ravel())

            Y[idx, :, 0] = h5file["firing_rate"][:-1].ravel() #* 10


    return X, Y


X, Y = get_dataset()



# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 2), return_sequences=True ))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model.fit(X, Y, epochs=50, batch_size=100, verbose=2)
model.save("pv_bas.keras")


Y_pred = model.predict(X)

t = np.linspace(0, 2000, 20000)
for idx in range(100):
    fig, axes = plt.subplots(nrows=2)

    axes[0].set_title(idx)
    axes[0].plot(t, Y_pred[idx, :, 0], label="LSTM", color="green")
    axes[0].plot(t, Y[idx, :, 0], label="Izhikevich model", color="red")

    axes[0].legend(loc="upper right")

    axes[1].plot(t, X[idx, :, 0], label="Ext conductance", color="orange")
    axes[1].plot(t, X[idx, :, 1], label="Inh conductance", color="blue")

    axes[1].legend(loc="upper right")

    plt.show(block=True)

    if idx > 20:
        break

