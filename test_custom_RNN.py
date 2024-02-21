import numpy as np
import tensorflow as tf
import keras
from keras.layers import Layer, Activation, Input, RNN
from keras import backend as K


class MyCell(Layer):
    def __init__(self, units, **kwargs):
        super(MyCell, self).__init__(**kwargs)

        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')


        self.built = True
    def call(self, inputs, states):

        #print(states[0])

        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)

        return output, output

    @property
    def state_size(self):
        return self.units

# Let's use this cell in a RNN layer:
cell = MyCell(2)
#iputs = Input((None, 2))  # Don't really understand the deal with None here.
rnn_layer = RNN(cell)
#outputs = rnn_layer(inputs)

model = keras.Sequential()
# model.add( Input( shape=(None, 2)) )
model.add(rnn_layer)
model.build(input_shape=(1, 2, 2))

x = np.random.rand(2, 2).reshape(1, 2, 2)
print(x)
outputs = model.predict( x )

print(outputs)