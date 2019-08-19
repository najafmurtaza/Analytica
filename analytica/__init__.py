__version__ = '0.0.1'

import numpy as np

class DataFrame:

	def __init__(self, data):
		"""
		Create 2D dataframe from dict

		Params
		------
		data: dict
			dict with str keys and numpy 1D array as values
		"""

		self._check_columns_type(data)
		self._check_columns_length(data)
		self._data = self._convert_unicode_to_object(data)
		self._dtypes_char = self._get_char_dtypes(data)

	def _check_columns_type(self, data):
		if not isinstance(data, dict):
			raise TypeError("Data must be dictionary")
		
		keys = data.keys()
		for single_key in keys:
			if not isinstance(single_key, str):
				raise TypeError("Keys must be of type str")
		
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

	def _get_char_dtypes(self, data):
		data_types_char = []
		for val in self._data.values():
			data_types_char.append(val.dtype.kind)
			
		return np.array(data_types_char, dtype='O')
		
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
				raise TypeError("column name must be of type 'str'")
		
		if len(columns) != len(set(columns)):
			raise ValueError("All columns names must be unique")

		self._data = dict(zip(columns, self._data.values()))

	def rename(self, cols_dict):
		"""
		Rename selected cols names

		Params
		------
		dict: key(old col), value(new col)

		Returns
		-------
		DataFrame: DataFrame with new cols names
		"""

		if not isinstance(cols_dict, dict):
			raise TypeError("New cols should be dict, mapping old to new col")

		old_cols = self.columns
		new_cols = cols_dict.keys()
		for col in new_cols:
			if col not in old_cols:
				raise ValueError(f"Column {col} doesn't exist in DataFrame")

		data = {}
		for col in old_cols:
			if col in new_cols:
				new_name = cols_dict[col]
				if not isinstance(new_name, str):
					raise TypeError(f"New column name {new_name} should be of type `str`")
				data[new_name] = self._data[col]
			else:
				data[col] = self._data[col]
		return DataFrame(data)

	@property
	def shape(self):
		"""
		Get no of rows and columns of dataframe

		Returns
		-------
		tuple: tuple of (Rows, cols)
		"""
		return (len(self), len(self.columns))

	@property
	def dtypes(self):
		"""
		Get types of all columns in dataframe

		Returns
		-------
		DataFrame: Two columns dataframe of cols and their types
		"""

		data_types = []
		for val in self._data.values():
			data_types.append(str(val.dtype))
		dtypes_dict = {'Column Name':np.array(self.columns), 'Data Type':np.array(data_types)}

		return DataFrame(dtypes_dict)

	@property
	def values(self):
		"""
		Get all column values as 2D array

		Returns
		-------
		ndarray: numpy 2d array of all column values in our dataframe
		"""
		# Casting to list becuase using iterables in stack will be deprecated after numpy 1.16
		return np.column_stack(list(self._data.values()))

	def _repr_html_(self):
		"""
		Used to create a string of HTML to nicely display the DataFrame
		in a Jupyter Notebook. Different string formatting is used for
		different data types.
		"""

		html = '<table><thead><tr><th></th>'
		for col in self.columns:
			html += f"<th>{col:10}</th>"

		html += '</tr></thead>'
		html += "<tbody>"

		only_head = False
		num_head = 10
		num_tail = 10
		if len(self) <= 20:
			only_head = True
			num_head = len(self)

		for i in range(num_head):
			html += f'<tr><td><strong>{i}</strong></td>'
			for col, values in self._data.items():
				kind = values.dtype.kind
				if kind == 'f':
					html += f'<td>{values[i]:10.3f}</td>'
				elif kind == 'b':
					html += f'<td>{values[i]}</td>'
				elif kind == 'O':
					v = values[i]
					if v is None:
						v = 'None'
					html += f'<td>{v:10}</td>'
				else:
					html += f'<td>{values[i]:10}</td>'
			html += '</tr>'

		if not only_head:
			html += '<tr><strong><td>...</td></strong>'
			for i in range(len(self.columns)):
				html += '<td>...</td>'
			html += '</tr>'
			for i in range(-num_tail, 0):
				html += f'<tr><td><strong>{len(self) + i}</strong></td>'
				for col, values in self._data.items():
					kind = values.dtype.kind
					if kind == 'f':
						html += f'<td>{values[i]:10.3f}</td>'
					elif kind == 'b':
						html += f'<td>{values[i]}</td>'
					elif kind == 'O':
						v = values[i]
						if v is None:
							v = 'None'
						html += f'<td>{v:10}</td>'
					else:
						html += f'<td>{values[i]:10}</td>'
				html += '</tr>'

		html += '</tbody></table>'

		return html

	def __getitem__(self, item):
		"""
		A python special function to get column values using bracket operator.

		Params
		------
		int: Returns selected row from all columns
		str: a column name [Returns all rows from one selected column]
		list: a list of column names [Returns all rows from list of columns]
				OR list of `ints` [Returns selected rows from all columns]
		slice(int): Returns selected rows from all columns
		DataFrame(Rows): DataFrame with boolean array for rows selction [Returns selected rows from all columns]
		tuple: Two valued tuple, one for rows and other for columns
				rows: `int` or `int list` or `slice` or `DataFrame`
				cols: `int` or `str` or `int/str/int+str list` or `int/str slice`

		Returns
		-------
		DataFrame: item column with values
		"""

		if isinstance(item, int):
			data = {}
			for col in self.columns:
				data[col] = self._data[col][[item]]
			return DataFrame(data)
		
		elif isinstance(item, str):
			return DataFrame({item:self._data[item]})

		elif isinstance(item, list):
			data = {}
			if isinstance(item[0], str):
				for col in item:
					data[col] = self._data[col]
			else:
				for key, val in self._data.items():
					data[key] = val[item]
			return DataFrame(data)

		elif isinstance(item, slice):
			data = {}
			for key, val in self._data.items():
				data[key] = val[item]
			return DataFrame(data)
			
		elif isinstance(item, DataFrame):
			if len(item.columns) != 1:
				raise ValueError("Only 1 column should be provided")
			bool_ind = item.values.flatten()
			if bool_ind.dtype.kind != 'b':
				raise TypeError("Values should be of type `bool`")
			data = {}
			for key, val in self._data.items():
				data[key] = val[bool_ind]
			return DataFrame(data)

		elif isinstance(item, tuple):
			if len(item) != 2:
				raise ValueError("Must pass two items, one for rows and other for cols")

			row_ind, col_ind = item

			if isinstance(row_ind, int):
				rows = [row_ind]
			elif isinstance(row_ind, DataFrame):
				if len(row_ind.columns) != 1:
					raise ValueError("Only 1 column should be provided")
				rows = row_ind.values.flatten()
				if rows.dtype.kind != 'b':
					raise TypeError("Values should be of type `bool`")
			elif isinstance(row_ind, (list, slice)):
				rows = row_ind
			else:
				raise TypeError("Rows index must be `int` or `int list` or `slice` or `DataFrame`")

			if isinstance(col_ind, int):
				cols = [self.columns[col_ind]]
			elif isinstance(col_ind, str):
				cols = [col_ind]
			elif isinstance(col_ind, list):
				cols = []
				columns = self.columns
				for ind in col_ind:
					if isinstance(ind, int):
						cols.append(columns[ind])
					elif isinstance(ind, str):
						cols.append(ind)
					else:
						raise TypeError("Columns can have either `str` or `int`")
			elif isinstance(col_ind, slice):
				start = col_ind.start
				stop = col_ind.stop
				step = col_ind.step

				columns = self.columns
				if isinstance(start, str) and isinstance(stop, str):
					start = columns.index(start)
					stop = columns.index(stop) + 1
				cols = columns[start:stop:step]
			else:
				raise TypeError("Cols index must be `int` or `str` or `int/str/int+str list` or `int/str slice`")

			data = {}
			for col in cols:
				data[col] = self._data[col][rows]
			
			return DataFrame(data)
		
		else:
			raise TypeError("Must pass `int` or `str` or `int/str list` `slice` or 'DataFrame` or `two items[row,col]`")

	def _aggregate_df(self, aggregate_func, axis=0,func_name=None, include_object=False, var_std=False):
		if axis == 0:
			data = {}
			for key, val in self._data.items():
				try:
					if var_std: 
						data[key] = np.array([aggregate_func(val, ddof=1)])
					else:
						data[key] = np.array([aggregate_func(val)])
				except:
					continue
			
			if len(data) > 0:
				return DataFrame(data)
			else:
				return []

		elif axis == 1:
			types_check = set(self._dtypes_char)
			if ((len(types_check) == 1) and (types_check.pop() == 'O'))  or include_object:
				arr = self.values
			else:
				new_df = {}
				for col in self.columns:
					val = self._data[col]
					if val.dtype.kind != 'O':
						new_df[col] = val
				new_df = DataFrame(new_df)
				arr = new_df.values
			try:
				if var_std:
					res = aggregate_func(arr, axis=axis, ddof=1)
				else:
					res = aggregate_func(arr, axis=axis)
			except:
				res = np.array([np.nan]*arr.shape[0])
			return DataFrame({func_name: res})
		
		else:
			raise ValueError("Axis can be either `0` or `1`")

	def min(self, axis=0):
		"""
		Get minimum from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Minimum values
		"""
		return self._aggregate_df(np.nanmin, axis, "min")

	def max(self, axis=0):
		"""
		Get maximum from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Maximum values
		"""
		return self._aggregate_df(np.nanmax, axis, "max")

	def mean(self, axis=0):
		"""
		Get mean from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Mean values
		"""
		return self._aggregate_df(np.nanmean, axis, 'mean')

	def median(self, axis=0):
		"""
		Get median from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Median values
		"""
		return self._aggregate_df(np.nanmedian, axis, "median")

	def sum(self, axis=0):
		"""
		Get sum from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Sum values
		"""
		return self._aggregate_df(np.nansum, axis, "sum")

	def argmax(self):
		"""
		Get max row indexes from DataFrame

		Returns
		-------
		DataFrame: Max row indexes
		"""
		return self._aggregate_df(np.nanargmax)

	def argmin(self):
		"""
		Get min row indexes from DataFrame
		
		Returns
		-------
		DataFrame: Min row indexes
		"""
		return self._aggregate_df(np.nanargmin)

	def _convert_to_proper(self, res, axis, key):
		"""
		As numpy's `all` and `all` are ambigous, sometimes they return `char` or `str`,
		So this internal function will convert to proper `bool` format.
		"""

		if axis == 0:
			for col, val in res._data.items():
				if val != None:
					res._data[col] = np.array([bool(val)])
				else:
					res._data[col] = np.array([True])
		elif axis == 1:
			values = next(iter(res._data.values()))
			bool_res = []
			for ind, val in enumerate(values):
				if val != None:
					bool_res.append(bool(val))
				else:
					bool_res.append(True)
			res._data[key] = np.array(bool_res)
			
		return res

	def all(self, axis=0):
		"""
		Check if all values are `true` or `false` from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Boolean values
		"""

		res = self._aggregate_df(np.all, axis, "all", True)
		return self._convert_to_proper(res, axis, 'all')

	def any(self, axis=0):
		"""
		Check if any value is `true` or `false` from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Boolean values
		"""

		res = self._aggregate_df(np.any, axis, "any", True)
		return self._convert_to_proper(res, axis, 'any')

	def var(self, axis=0):
		"""
		Get variance from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Variance values
		"""
		return self._aggregate_df(np.nanvar, axis, "var", var_std=True)

	def std(self, axis=0):
		"""
		Get Std. Dev from DataFrame rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Std. Dev values
		"""
		return self._aggregate_df(np.nanstd, axis, "std", var_std=True)

	def isna(self):
		"""
		Check all values of DataFrame whether they are `None` or `NaN`

		Returns
		-------
		DataFrame: Same shape as of original but boolean values
		"""

		new_df = {}
		for col in self.columns:
			val = self._data[col]
			if val.dtype.kind != 'O':
				new_df[col] = np.isnan(val)
			else:
				new_df[col] = (val==None)

		return DataFrame(new_df)

	def head(self, n=5):
		"""
		Get first `n` rows from DataFrame

		Params
		------
		int: No. of rows to get
		
		Returns
		-------
		DataFrame: First `n` rows from all columns
		"""
		return self[:n]

	def tail(self, n=5):
		"""
		Get last `n` rows from DataFrame

		Params
		------
		int: No. of rows to get
		
		Returns
		-------
		DataFrame: Last `n` rows from all columns
		"""
		return self[-n:]

	def count(self, axis=0):
		"""
		Get count of non-missing values from rows or cols

		Params
		------
		int: 0 for row wise [Default]
			 1 for column wise

		Returns
		-------
		DataFrame: Counts
		"""

		bool_df = self.isna()
		if axis == 0:
			data = {}
			for col, val in bool_df._data.items():
				data[col] = np.array([np.sum(~val)])
			return DataFrame(data)

		elif axis == 1:
			arr = ~bool_df.values
			res = np.sum(arr, axis=1)
			return DataFrame({"count": res})
		
		else:
			raise ValueError("Axis can be either `0` or `1`")