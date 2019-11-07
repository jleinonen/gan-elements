from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras import backend as K


class GradientPenalty(Layer):
    def call(self, inputs):
        target, wrt = inputs
        grad = K.gradients(target, wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), 
            axis=1, keepdims=True))-1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        return x*weights + y*(1-weights)
