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

	@columns.setter
	def columns(self, columns):
		"""
		Set columns name of dataframe

		Params
		------
		list: A list of new columns names with same length as of original
		"""

		if not isinstance(columns, list):
			raise TypeError("columns must be of type 'list'")

		if len(columns) != len(self._data.keys()):
			raise ValueError("new columns must have same length as original columns")

		for i in columns:
			if not isinstance(i, str):
				raise TypeError("column name must be of type 'string'")
		
		if len(columns) != len(set(columns)):
			raise ValueError("All columns names must be unique")

		self._data = dict(zip(columns, self._data.values()))