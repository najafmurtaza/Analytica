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
    
    def test_rename(self):
        df_result = df.rename({'a': 'A', 'c': 'C'})
        df_answer = alt.DataFrame({'A': a, 'b': b, 'C': c, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df.rename({'w': 'A', 'c': 'C'})
        
        with pytest.raises(TypeError):
            df.rename({'a': 'A', 'c': 1})

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

    def test_list_columns(self):
        df_answer = alt.DataFrame({'c': c, 'e': e})
        assert_df_equals(df[:, [2, 4]], df_answer)
        assert_df_equals(df[:, [2, 'e']], df_answer)
        assert_df_equals(df[:, ['c', 'e']], df_answer)

        df_result = df[2, ['a', 'e']]
        df_answer = alt.DataFrame({'a': a[[2]], 'e': e[[2]]})
        assert_df_equals(df_result, df_answer)

        df_answer = alt.DataFrame({'c': c[[1, 2]], 'e': e[[1, 2]]})
        assert_df_equals(df[[1, 2], ['c', 'e']], df_answer)

        df1 = alt.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        df_answer = alt.DataFrame({'c': c[[0, 2]], 'e': e[[0, 2]]})
        assert_df_equals(df[df1['a'], ['c', 'e']], df_answer)

    def test_col_slice(self):
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': c})
        assert_df_equals(df[:, :3], df_answer)

        df_answer = alt.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2]})
        assert_df_equals(df[::2, :3], df_answer)

        df_answer = alt.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2], 'd': d[::2], 'e': e[::2]})
        assert_df_equals(df[::2, :], df_answer)

        with pytest.raises(TypeError):
            df[:, set()]

    def test_new_column(self):
        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = np.array([1.5, 23, 4.11])
        df_result['f'] = f
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result['f'] = True
        f = np.repeat(True, 3)
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = np.array([1.5, 23, 4.11])
        df_result['c'] = f
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': f, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = [1.5, 23, 4.11]
        df_result['f'] = f
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': np.array(f)})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result[1,'a'] = 'f'
        w = a.copy()
        w[1] = 'f'
        df_answer = alt.DataFrame({'a': w, 'b': b, 'c': c, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result[[1,2],'c'] = 1.2
        w = c.copy()
        w[[1,2]] = 1.2
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': w, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        df_result = alt.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result[1:3,'c'] = 1.2
        w = c.copy()
        w[[1,2]] = 1.2
        df_answer = alt.DataFrame({'a': a, 'b': b, 'c': w, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(TypeError):
            df[1] = 5
        with pytest.raises(TypeError):
            df['a'] = df
        with pytest.raises(ValueError):
            df['a'] = [5,8]
        with pytest.raises(ValueError):
            df['a'] = [[5,8,6],[5,8,8],[5,8,8]]
        with pytest.raises(ValueError):
            df['a'] = np.array([[5,8,6],[5,8,8],[5,8,8]])
        with pytest.raises(ValueError):
            df['a'] = np.array([5,6])
        with pytest.raises(TypeError):
            df[1,df] = 6
        with pytest.raises(TypeError):
            df[df,'a'] = 'v'
        with pytest.raises(TypeError):
            df[2,1] = 6

a5 = np.array([11, 5])
b5 = np.array([3.4, 5.1])
df5 = alt.DataFrame({'a': a5, 'b': b5})

class TestOperators:

    def test_add(self):
        df_result = df5 + 3
        df_answer = alt.DataFrame({'a': a5 + 3, 'b': b5 + 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 + df5
        assert_df_equals(df_result, df_answer)

    def test_sub(self):
        df_result = df5 - 3
        df_answer = alt.DataFrame({'a': a5 - 3, 'b': b5 - 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 - df5
        df_answer = alt.DataFrame({'a': 3 - a5, 'b': 3 - b5})
        assert_df_equals(df_result, df_answer)

    def test_mul(self):
        df_result = df5 * 3
        df_answer = alt.DataFrame({'a': a5 * 3, 'b': b5 * 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 * df5
        assert_df_equals(df_result, df_answer)

    def test_truediv(self):
        df_result = df5 / 3
        df_answer = alt.DataFrame({'a': a5 / 3, 'b': b5 / 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 / df5
        df_answer = alt.DataFrame({'a': 3 / a5, 'b': 3 / b5})
        assert_df_equals(df_result, df_answer)

    def test_floordiv(self):
        df_result = df5 // 3
        df_answer = alt.DataFrame({'a': a5 // 3, 'b': b5 // 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 // df5
        df_answer = alt.DataFrame({'a': 3 // a5, 'b': 3 // b5})
        assert_df_equals(df_result, df_answer)

    def test_pow(self):
        df_result = df5 ** 3
        df_answer = alt.DataFrame({'a': a5 ** 3, 'b': b5 ** 3})
        assert_df_equals(df_result, df_answer)

        df_result = 2 ** df5
        df_answer = alt.DataFrame({'a': 2 ** a5, 'b': 2 ** b5})
        assert_df_equals(df_result, df_answer)

    def test_gt_lt(self):
        df_result = df5 > 3
        df_answer = alt.DataFrame({'a': a5 > 3, 'b': b5 > 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 < 2
        df_answer = alt.DataFrame({'a': a5 < 2, 'b': b5 < 2})
        assert_df_equals(df_result, df_answer)

    def test_ge_le(self):
        df_result = df5 >= 3
        df_answer = alt.DataFrame({'a': a5 >= 3, 'b': b5 >= 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 < 2
        df_answer = alt.DataFrame({'a': a5 <= 2, 'b': b5 <= 2})
        assert_df_equals(df_result, df_answer)

    def test_eq_ne(self):
        df_result = df5 == 3
        df_answer = alt.DataFrame({'a': a5 == 3, 'b': b5 == 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 != 2
        df_answer = alt.DataFrame({'a': a5 != 2, 'b': b5 != 2})
        assert_df_equals(df_result, df_answer)

a1 = np.array(['a', 'b', 'c'])
b1 = np.array([11, 5, 8])
c1 = np.array([3.4, np.nan, 5.1])
df1 = alt.DataFrame({'a': a1, 'b': b1, 'c': c1})

class TestAggregation:

    def test_min(self):
        df_result = df1.min()
        df_answer = alt.DataFrame({'a': np.array(['a'], dtype='O'),
                                   'b': np.array([5]),
                                   'c': np.array([3.4])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.min(axis=1)
        df_answer = alt.DataFrame({'min': np.array([3.4, 5, 5.1])})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df1.min(axis=5)
        with pytest.raises(ValueError):
            df1.min(axis='1')

    def test_max(self):
        df_result = df1.max()
        df_answer = alt.DataFrame({'a': np.array(['c'], dtype='O'),
                                   'b': np.array([11]),
                                   'c': np.array([5.1])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.max(axis=1)
        df_answer = alt.DataFrame({'max': np.array([11, 5, 8])})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df1.max(axis=5)
        with pytest.raises(ValueError):
            df1.max(axis='1')

    def test_mean(self):
        df_result = df1.mean()
        df_answer = alt.DataFrame({'b': np.array([8.]),
                                   'c': np.array([4.25])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.mean(axis=1)
        df_answer = alt.DataFrame({'mean': np.array([7.2, 5.0, 6.55])})
        assert_df_equals(df_result, df_answer)

    def test_median(self):
        df_result = df1.median()
        df_answer = alt.DataFrame({'b': np.array([8]),
                                   'c': np.array([4.25])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.median(axis=1)
        df_answer = alt.DataFrame({'median': np.array([7.2, 5.0, 6.55])})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        df_result = df1.sum()
        df_answer = alt.DataFrame({'a': np.array(['abc'], dtype='O'),
                                   'b': np.array([24]),
                                   'c': np.array([8.5])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.sum(axis=1)
        df_answer = alt.DataFrame({'sum': np.array([14.4, 5.0, 13.1])})
        assert_df_equals(df_result, df_answer)

    def test_argmax(self):
        df_result = df1.argmax()
        df_answer = alt.DataFrame({'a': np.array([2]),
                                   'b': np.array([0]),
                                   'c': np.array([2])})
        assert_df_equals(df_result, df_answer)

    def test_argmin(self):
        df_result = df1.argmin()
        df_answer = alt.DataFrame({'a': np.array([0]),
                                   'b': np.array([1]),
                                   'c': np.array([0])})
        assert_df_equals(df_result, df_answer)

    def test_all(self):
        df_result = df1.all()
        df_answer = alt.DataFrame({'a': np.array([True]),
                                   'b': np.array([True]),
                                   'c': np.array([True])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.all(axis=1)
        df_answer = alt.DataFrame({'all': np.array([True, True, True])})
        assert_df_equals(df_result, df_answer)

    def test_any(self):
        df_result = df1.any()
        df_answer = alt.DataFrame({'a': np.array([True]),
                                   'b': np.array([True]),
                                   'c': np.array([True])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.any(axis=1)
        df_answer = alt.DataFrame({'any': np.array([True, True, True])})
        assert_df_equals(df_result, df_answer)

    def test_var(self):
        df_result = df1.var()
        df_answer = alt.DataFrame({'b': np.array([np.nanvar(b1, ddof=1)]),
                                   'c': np.array([np.nanvar(c1, ddof=1)])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.var(axis=1)
        df_answer = alt.DataFrame({'var': np.array([28.88, np.nan, 4.205])})
        assert_df_equals(df_result, df_answer)

    def test_std(self):
        df_result = df1.std()
        df_answer = alt.DataFrame({'b': np.array([np.nanstd(b1, ddof=1)]),
                                   'c': np.array([np.nanstd(c1, ddof=1)])})
        assert_df_equals(df_result, df_answer)

        df_result = df1.std(axis=1)
        df_answer = alt.DataFrame({'std': np.array([5.37401154, np.nan, 2.05060967])})
        assert_df_equals(df_result, df_answer)

df3 = alt.DataFrame({'a':a, 'b':b, 'c':c1})
class TestOtherMethods:

    def test_isna(self):
        df_result = df3.isna()
        df_answer = alt.DataFrame({'a': np.array([False, False, False]),
                                   'b': np.array([False, False, True]),
                                   'c': np.array([False, True, False])})
        assert_df_equals(df_result, df_answer)

    def test_head(self):
        df_result = df.head(2)
        df_answer = alt.DataFrame({'a': a[:2], 'b': b[:2], 'c': c[:2],
                                   'd': d[:2], 'e': e[:2]})
        assert_df_equals(df_result, df_answer)

    def test_tail(self):
        df_result = df.tail(2)
        df_answer = alt.DataFrame({'a': a[-2:], 'b': b[-2:], 'c': c[-2:],
                                   'd':d[-2:], 'e': e[-2:]})
        assert_df_equals(df_result, df_answer)

    def test_count(self):
        df_result = df3.count()
        df_answer = alt.DataFrame({'a': np.array([3]),
                                   'b': np.array([2]),
                                   'c': np.array([2])})
        assert_df_equals(df_result, df_answer)

        df_result = df3.count(axis=1)
        df_answer = alt.DataFrame({'count': np.array([3,2,2])})
        assert_df_equals(df_result, df_answer)

    def test_csv(self):
        df_result = alt.read_csv('./data/PizzaData.csv')
        assert df_result.columns == ['id','address','categories','city','country','keys','latitude','longitude','menuPageURL','menus.amountMax','menus.amountMin','menus.currency','menus.dateSeen','menus.description','menus.name','name','postalCode','priceRangeCurrency','priceRangeMin','priceRangeMax','province']
        assert df_result.shape == (3510,21)

        df_result = alt.read_csv('./data/PizzaData.csv', header=False)
        assert df_result.shape == (3511,21)
        
        with pytest.raises(TypeError):
            alt.read_csv('./data/PizzaData.csv', header=45)

    def test_sample(self):
        df_result = df3.sample(2, seed=1)
        df_answer = alt.DataFrame({'a': np.array(['a', 'c'], dtype=object),
                                   'b': np.array(['c', None]),
                                   'c': np.array([3.4, 5.1])})
        assert_df_equals(df_result, df_answer)

        df_result = df3.sample(frac=.7, seed=1)
        df_answer = alt.DataFrame({'a': np.array(['a', 'c', 'b'], dtype=object),
                                   'b': np.array(['c', None, 'd']),
                                   'c': np.array([3.4, 5.1, np.nan])})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(TypeError):
            df.sample(2.5)

        with pytest.raises(ValueError):
            df.sample(frac=-2)
