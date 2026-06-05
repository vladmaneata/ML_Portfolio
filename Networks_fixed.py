import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from EllipticCurvesExtended import EllipticCurve

# Set up crypto parameters
m_bits = 16
a, b, p = 1, 1, 23  # Curve parameters
curve = EllipticCurve(a, b, p)

puk_bits = curve.get_key_shape()[1]  # public key bits
prk_bits = curve.get_key_shape()[0]  # private key bits
c_bits = (m_bits + puk_bits) // 2
pad = 'same'

# Size of the message space
m_train = 2**(m_bits)

# Alice Network
ainput0 = Input(shape=(m_bits,), name="message")        # Message input
ainput1 = Input(shape=(puk_bits,), name="public_key")   # Public key input
ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + puk_bits), activation='tanh')(ainput)
areshape = Reshape((m_bits + puk_bits, 1))(adense1)
aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad, activation='tanh')(areshape)
aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad, activation='tanh')(aconv1)
aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad, activation='tanh')(aconv2)
aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad, activation='sigmoid')(aconv3)
aoutput = Flatten(name="ciphertext")(aconv4)

alice = Model(inputs=[ainput0, ainput1], outputs=aoutput, name='alice')

# Bob Network
binput0 = Input(shape=(c_bits,), name="ciphertext_input")
binput1 = Input(shape=(prk_bits,), name="private_key")
binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=(m_bits * 2), activation='tanh')(binput)
breshape = Reshape((m_bits * 2, 1))(bdense1)
bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad, activation='tanh')(breshape)
bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad, activation='tanh')(bconv1)
bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad, activation='tanh')(bconv2)
bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad, activation='sigmoid')(bconv3)
boutput = Flatten(name="recovered_message")(bconv4)

bob = Model(inputs=[binput0, binput1], outputs=boutput, name='bob')

# Eve Network
einput = Input(shape=(c_bits,), name="eve_ciphertext_input")
edense1 = Dense(units=(m_bits * 2), activation='tanh')(einput)
edense2 = Dense(units=(m_bits * 2), activation='tanh')(edense1)
ereshape = Reshape((m_bits * 2, 1))(edense2)
econv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad, activation='tanh')(ereshape)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad, activation='tanh')(econv1)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad, activation='tanh')(econv2)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad, activation='sigmoid')(econv3)
eoutput = Flatten(name="eve_guess")(econv4)

eve = Model(inputs=einput, outputs=eoutput, name='eve')

# Outputs for loss
aliceout = alice([ainput0, ainput1])
bobout = bob([aliceout, binput1])
eveout = eve(aliceout)

# Custom ABE loss
def abe_custom_loss(ainput0, bobout, eveout):
    def loss_fn(y_true, y_pred):
        bob_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ainput0 - bobout), axis=-1))
        eve_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ainput0 - eveout), axis=-1))
        return bob_loss + tf.square(m_bits / 2 - eve_loss) / ((m_bits // 2) ** 2)
    return loss_fn

# Custom Eve loss
def eve_custom_loss(ainput0, eveout):
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(ainput0 - eveout), axis=-1))
    return loss_fn

# Optimizers
beoptim = Adam(learning_rate=0.0001)
eveoptim = Adam(learning_rate=0.0001)

# ABE model (for Alice & Bob training)
abemodel = Model(inputs=[ainput0, ainput1, binput1], outputs=[bobout,eveout], name='abemodel')
abemodel.compile(optimizer=beoptim, loss=abe_custom_loss(ainput0, bobout, eveout))

# Eve model (Alice is frozen)
alice.trainable = False
evemodel = Model(inputs=[ainput0, ainput1], outputs=eveout, name='evemodel')
evemodel.compile(optimizer=eveoptim, loss=eve_custom_loss(ainput0, eveout))