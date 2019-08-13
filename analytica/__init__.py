__version__ = '0.0.1'

import numpy as np

class DataFrame:

	def __init__(self, data):
		"""
		Create 2D dataframe from dict

		Params
		------
		data: dict
			dict with string keys and numpy 1D array as values
		"""

		self._check_columns_type(data)
		self._check_columns_length(data)
		self._data = self._convert_unicode_to_object(data)

	def _check_columns_type(self, data):
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

	def _check_columns_length(self, data):
		values = data.values()
		length = len(next(iter(values)))
		for index, single_val in enumerate(values):
			if len(single_val) != length:
				raise ValueError("Length of columns should be same")

	def _convert_unicode_to_object(self, data):
		converted_data = {}
		for key, val in data.items():
			if val.dtype.kind == 'U':
				converted_data[key] = data[key].astype('O')
			else:
				converted_data[key] = data[key]

		return converted_data

	def __len__(self):
		"""
		An implementation of python special function.
		Count no of rows of dataframe

		Returns
		------
		int: Total no. of rows in our dataframe
		"""
		values = self._data.values()
		length = len(next(iter(values)))

		return length

	@property 
	def columns(self):
		"""
		Property method to get columns names as list

		Returns
		-------
		list: columns names as list
		"""
		
		return list(self._data.keys())