__version__ = '0.0.1'

import numpy as np

class DataFrame:

	def __init__(self, data):
		self._check_column_types(data)

	def _check_column_types(self, data):
		if not isinstance(data, dict):
			raise TypeError("Data must be dictionary")
		
		keys = data.keys()
		for single_key in keys:
			if not isinstance(single_key, str):
				raise TypeError("Keys must be of type string")
		
		values = data.values()
		for single_val in values:
			if not isinstance(single_val, np.ndarray):
				raise TypeError("Values must be numpy 1D arrays")
			if single_val.ndim != 1:
				raise ValueError("Arrays dim must be equal to 1")
