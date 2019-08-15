import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
import analytica as alt

def assert_df_equals(df1, df2):
    assert df1.columns == df2.columns
    for values1, values2 in zip(df1._data.values(), df2._data.values()):
        kind = values1.dtype.kind
        if kind == 'f':
            assert_allclose(values1, values2)
        else:
            assert_array_equal(values1, values2)

a = np.array(['a', 'b', 'c'])
b = np.array(['c', 'd', None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])
df = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})

class TestDataFrameCreation:

    def test_input_types(self):
        with pytest.raises(TypeError):
            alt.DataFrame([1, 2, 3])

        with pytest.raises(TypeError):
            alt.DataFrame({1: 5, 'b': 10})

        with pytest.raises(TypeError):
            alt.DataFrame({'a': np.array([1]), 'b': 10})

        with pytest.raises(ValueError):
            alt.DataFrame({'a': np.array([1]), 
                           'b': np.array([[1]])})

        # correct construction. no error
        alt.DataFrame({'a': np.array([1]), 
                       'b': np.array([1])})
    
    def test_array_length(self):
        with pytest.raises(ValueError):
            alt.DataFrame({'a': np.array([1, 2]), 
                           'b': np.array([1])})
        # correct construction. no error                           
        alt.DataFrame({'a': np.array([1, 2]), 
                        'b': np.array([5, 10])})
            
    def test_unicode_to_object(self):
        a_object = a.astype('O')
        assert df._data['a'].dtype.kind == a_object.dtype.kind
        assert df._data['b'].dtype.kind == b.dtype.kind
        assert df._data['c'].dtype.kind == c.dtype.kind
        assert df._data['d'].dtype.kind == d.dtype.kind
        assert df._data['e'].dtype.kind == e.dtype.kind

    def test_len(self):
        assert len(df) == 3

    def test_columns(self):
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_set_columns(self):
        with pytest.raises(TypeError):
            df.columns = 5

        with pytest.raises(ValueError):
            df.columns = ['a', 'b']

        with pytest.raises(TypeError):
            df.columns = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError):
            df.columns = ['f', 'f', 'g', 'h', 'i']

        df.columns = ['f', 'g', 'h', 'i', 'j']
        assert df.columns == ['f', 'g', 'h', 'i', 'j']

        # set it back
        df.columns = ['a', 'b', 'c', 'd', 'e']
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_shape(self):
        assert df.shape == (3, 5)
    
    def test_dtypes(self):
        cols = np.array(['a', 'b', 'c', 'd', 'e'], dtype='O')
        dtypes = np.array([str(a.astype('O').dtype), str(b.dtype),
                            str(c.dtype), str(d.dtype), str(e.dtype)], dtype='O')
        
        df_answer = alt.DataFrame({'Column Name': cols,
                                    'Data Type': dtypes})
        df_result = df.dtypes
        assert_df_equals(df_result, df_answer)
    
    def test_values(self):
        values = np.column_stack((a, b, c, d, e))
        assert_array_equal(df.values, values)

class TestSelection:

    def test_one_column(self):
        assert_array_equal(df['a'].values[:, 0], a)
        assert_array_equal(df['c'].values[:, 0], c)
    
    def test_multiple_columns(self):
        cols = ['a', 'c']
        df_result = df[cols]
        df_answer = alt.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_simple_boolean(self):
        bool_arr = np.array([True, False, False])
        df_bool = alt.DataFrame({'col': bool_arr})
        df_result = df[df_bool]
        df_answer = alt.DataFrame({'a': a[bool_arr], 'b': b[bool_arr], 
                                   'c': c[bool_arr], 'd': d[bool_arr], 
                                   'e': e[bool_arr]})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df_bool = alt.DataFrame({'col': bool_arr, 'col2': bool_arr})
            df[df_bool]

        with pytest.raises(TypeError):
            df_bool = alt.DataFrame({'col': np.array([1, 2, 3])})
            df[df_bool]
    
    def test_simultaneous_tuple(self):
        with pytest.raises(TypeError):
            s = set()
            df[s]

        with pytest.raises(ValueError):
            df[1, 2, 3]
    
    def test_single_element(self):
        df_answer = alt.DataFrame({'e': np.array([2])})
        assert_df_equals(df[1, 'e'], df_answer)
    
    def test_all_row_selections(self):
        df1 = alt.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        with pytest.raises(ValueError):
            df[df1, 'e']

        with pytest.raises(TypeError):
            df[df1['b'], 'c']

        df_result = df[df1['a'], 'c']
        df_answer = alt.DataFrame({'c': c[[True, False, True]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[[1, 2], 0]
        df_answer = alt.DataFrame({'a': a[[1, 2]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[1:, 0]
        assert_df_equals(df_result, df_answer)