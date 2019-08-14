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
        dtypes = np.array(['O', 'O', 'f', 'b', 'i'], dtype='O')

        df_result = df.dtypes
        df_answer = alt.DataFrame({'Column Name': cols,
                                    'Data Type': dtypes})
        assert_df_equals(df_result, df_answer)