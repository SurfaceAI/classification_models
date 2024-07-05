import math
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions


class CLM(tf.keras.layers.Layer):
	"""
	Proportional Odds Model activation layer.
	"""

	def __init__(self, num_classes, link_function, min_distance=0.35, name=None, use_slope=False, fixed_thresholds=False, **kwargs):
		self.num_classes = num_classes
		self.dist = distributions.Normal(loc=0., scale=1.)
		self.link_function = link_function
		self.min_distance = min_distance
		self.use_slope = use_slope
		self.fixed_thresholds = fixed_thresholds
		super(CLM, self).__init__(**kwargs, name=name)

	def _convert_thresholds(self, b, a, min_distance=0.35):
		a = tf.pow(a, 2)
		a = a + min_distance
		thresholds_param = tf.cast(tf.concat([b, a], axis=0), dtype='float32')
		th = tf.reduce_sum(
			tf.linalg.band_part(tf.ones([self.num_classes - 1, self.num_classes - 1]), -1, 0) * tf.reshape(
				tf.tile(thresholds_param, [self.num_classes - 1]), shape=[self.num_classes - 1, self.num_classes - 1]),
			axis=1)
		return th

	@tf.autograph.experimental.do_not_convert
	def _nnpom(self, projected, thresholds):
		projected = tf.cast(tf.reshape(projected, shape=[-1]), dtype='float32')

		if self.use_slope and self.slope:
			projected = projected * self.slope
			thresholds = thresholds * self.slope

		m = tf.shape(projected)[0]
		a = tf.reshape(tf.tile(thresholds, [m]), shape=[m, -1])
		b = tf.transpose(tf.reshape(tf.tile(projected, [self.num_classes - 1]), shape=[-1, m]))
		z3 = a - b


		if self.link_function == 'probit':
			a3T = self.dist.cdf(z3)
		elif self.link_function == 'cloglog':
			a3T = 1 - tf.exp(-tf.exp(z3))
		else: # logit
			a3T = 1.0 / (1.0 + tf.exp(-z3))

		ones = tf.ones(tf.convert_to_tensor([m, 1], dtype=tf.int32))
		a3 = tf.concat([a3T, ones], axis=1)
		a3 = tf.concat([tf.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, 0:-1]], axis=-1)

		return a3

	def build(self, input_shape):
		if not self.fixed_thresholds:
			self.thresholds_b = self.add_weight('b_b_nnpom', shape=(1,),
												initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0.1))
			self.thresholds_a = self.add_weight('b_a_nnpom', shape=(self.num_classes - 2,),
												initializer=tf.keras.initializers.RandomUniform(
													minval=math.sqrt((1.0 / (self.num_classes - 2)) / 2),
													maxval=math.sqrt(1.0 / (self.num_classes - 2))))

		if self.use_slope:
			self.slope = self.add_weight('slope', shape=(1,), initializer=tf.keras.initializers.Constant(100.0))


	def call(self, x, **kwargs):
		if self.fixed_thresholds:
			# thresholds = self.fixed_thresholds
			thresholds = np.linspace(0, 1, 11, dtype=np.float32)[1:-1]
			# thresholds = np.array([0.09090909, 0.27272727, 0.36363636, 0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091], dtype=np.float32)
		else:
			thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a, self.min_distance)

		return self._nnpom(x, thresholds)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)

	def get_config(self):
		base_config = super(CLM, self).get_config()
		base_config['num_classes'] = self.num_classes
		base_config['link_function'] = self.link_function
		base_config['min_distance'] = self.min_distance
		base_config['use_slope'] = self.use_slope
		base_config['fixed_thresholds'] = self.fixed_thresholds

		return base_config
		