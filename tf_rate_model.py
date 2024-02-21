import numpy as np
import tensorflow as tf
import keras
from keras import layers
#from keras.layers import Layer, Activation, Input, AbstractRNNCell
from keras import backend as K
import json
class RateModel(layers.Layer):

    def __init__(self, params, dt=0.1, **kwargs):
        super(RateModel, self).__init__(**kwargs)
        self.dt = tf.convert_to_tensor(dt)
        self.MaxFR = tf.convert_to_tensor( params['MaxFR'] )
        self.Sfr = tf.convert_to_tensor( params['Sfr'] )
        self.th =  tf.convert_to_tensor( params['th'] )

        self.r = tf.convert_to_tensor( params['r'] )
        self.q = tf.convert_to_tensor( params['q'] )
        self.s = tf.convert_to_tensor( params['s'] )

        self.tau_FR = tf.convert_to_tensor( params['tau_FR'] )
        self.tau_A = tf.convert_to_tensor( params['tau_A'] )
        self.winh = tf.convert_to_tensor( params['winh'] )

        # self.tau_f = tf.convert_to_tensor( kwargs['tau_f'] )
        # self.tau_d = tf.convert_to_tensor( kwargs['tau_f'] )
        # self.tau_r = tf.convert_to_tensor( kwargs['tau_r'] )
        # self.Uinc  = tf.convert_to_tensor( kwargs['Uinc'] )
        # self.gsyn_max = tf.convert_to_tensor( kwargs['gsyn_max'] )
        # self.pconn = tf.convert_to_tensor( kwargs['pconn'] )

        self.units = tf.size(self.MaxFR)
        self.state_size = [self.units, self.units]
        # #self.ninp = tf.shape(self.gsyn_max)[0] - self.units
        #
        self.exp_tau_FR = K.exp(-self.dt / self.tau_FR)
        self.exp_tau_A = K.exp(-self.dt / self.tau_A)

    def build(self, input_shape):
        super().build(input_shape)
    #     # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
    #     #                               initializer='uniform',
    #     #                               name='kernel')
    #     # self.recurrent_kernel = self.add_weight(
    #     #     shape=(self.units, self.units),
    #     #     initializer='uniform',
    #     #     name='recurrent_kernel')
        self.built = True

    def I_F_cuve(self, x):
        shx_2 = (x - self.th)**2

        y = tf.nn.relu(x - self.th) * self.MaxFR * (x - self.th) / (self.Sfr + shx_2)
        # y = tf.where(x > self.th, self.MaxFR * shx_2 / (self.Sfr + shx_2), 0)
        return y
    def call(self, inputs, states):
        # prev_output = states[0]
        # h = K.dot(inputs, self.kernel)
        # output = h + K.dot(prev_output, self.recurrent_kernel)

        FR = states[0] # [0,  : self.units]
        Ad = states[1] #[0, self.units : ]

        R = states[2]
        U = states[3]
        X = states[3]


        FRpre_normed = self.pconn * K.concat(FR, inputs)

        y_ = R * K.exp(-self.dt / self.tau_d)

        x_ = 1 + (X - 1 + self.tau1r * U) * K.exp(-self.dt / self.tau_r) - self.tau1r * U

        u_ = U * K.exp(-self.dt / self.tau_f)
        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        R = y_ + U * x_ * FRpre_normed
        X = x_ - U * x_ * FRpre_normed

        Isyn = self.gmax * X


        FR_inf = (1 - self.r * FR) * self.I_F_cuve(Isyn - self.q * Ad)
        A_inf = self.s * FR

        FR = FR_inf - (FR_inf - FR) * self.exp_tau_FR
        Ad = A_inf - (A_inf - Ad) * self.exp_tau_A


        return FR, [FR, Ad, R, U, X]
#################################################
Nunits = 2
with open("optim_res.json", "r") as outfile:
    params = json.load(outfile)
for key, val in params.items():
    params[key] = np.zeros(Nunits, dtype=np.float32) + val

input_shape = (1, 20, Nunits)

model = keras.Sequential()
model.add( layers.RNN( RateModel(params, dt=0.1), return_sequences=True ))

model.build( input_shape=input_shape)

#initial_state = [tf.zeros((batch_size, units)), tf.zeros((batch_size, units))]
#outputs, state = model(inputs, initial_state=initial_state)
X = np.zeros(input_shape, dtype=np.float32)
print(X)
Y = model.predict(X)

print(Y)
#model.summary()
