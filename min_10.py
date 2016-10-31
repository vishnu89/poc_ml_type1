from lib_collection.libs import *


@stop_print
def init():
    train = r"D:\intel\intel\webCrawler\twitter_ana\uint_7\Mining-the-Social-Web-2nd-Edition-master\ipynb\machine_learning\loan_1\train.csv"
    test = r"D:\intel\intel\webCrawler\twitter_ana\uint_7\Mining-the-Social-Web-2nd-Edition-master\ipynb\machine_learning\loan_1\test.csv"
    test = pd.read_csv(test,
                       # skiprows = [0],
                       names=["id", 'age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship',
                              'race', 'sex', 'hrprwk', 'native_country'],
                       # index_col="id",
                       nrows=None,
                       skip_blank_lines=True,
                       encoding=None,
                       error_bad_lines=False,  # default true (Only valid with C parser)
                       warn_bad_lines=True,  # valid only if error_bad_lines = False
                       header=0,
                       engine='c'
                       )
    train = pd.read_csv(train,
                        # skiprows = [0],
                        names=["id", 'age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship',
                               'race', 'sex', 'hrprwk', 'native_country', 'income_group'],
                        # index_col="id",
                        nrows=None,
                        skip_blank_lines=True,
                        encoding=None,
                        error_bad_lines=False,  # default true (Only valid with C parser)
                        warn_bad_lines=True,  # valid only if error_bad_lines = False
                        header=0,
                        engine='c'
                        )
    print train.columns.tolist()
    return test, train


@stop_print
def init_fill(test,train):
    print drop_values(train.apply(num_missing), 0)
    train = train.apply(lambda x: x.fillna(x.mode()[0]) if x.isnull().any() else x)
    print drop_values(train.apply(num_missing), 0)
    return test, train


@stop_print
def init_transform(test,train,threshold=.05):
    categorical_atr = get_atr_by_type(train, 'object')
    update_vals = train[categorical_atr].apply(group_small_vals, threshold=threshold)
    # print update_vals
    check = train.apply(count_class).to_frame('old_train')
    check.insert(0, 'old_test', test.apply(count_class))
    for i, k in update_vals.iteritems():
        if k and i != 'income_group':
            print threshold, i, k
            train[i].replace(k, inplace=True)
            test[i].replace(k, inplace=True)
    check.insert(0, 'new_train', train.apply(count_class))
    check.insert(0, 'new_test', test.apply(count_class))
    print check[['old_train', 'new_train', 'old_test', 'new_test']]
    return test, train


def num_missing(x): return x.isnull().sum()


def count_class(x): return len(x.unique())


def get_atr_by_type(x, y): return x.dtypes[x.dtypes == y].index.tolist()


@stop_print
def normalizer(*ser):
    if len(ser) == 1:
        result = ser[0].value_counts() / ser[0].shape
    else:
        result = {x.name: x.value_counts() / x.shape for x in ser}
    print result
    return result


def drop_values(ser, *vals):
    for val in vals:
        ser = ser[ser != val]
    return ser


@stop_print
def group_small_vals(col, threshold, new_group_name='others'):
    n_col = normalizer(col)
    small_groups = n_col[n_col <= threshold].index
    new_groups = {k: new_group_name for k in small_groups if k}
    print n_col
    return new_groups





if __name__ == '__main__':
    pass
    # temp = init_fill(prints=True)
    # print train.dtypes
    # print train.describe()
    # print train.quantile([.25,.75]).diff().ix[.75]
    # print train[get_dtype(train,'object')].apply(lambda x: len(x.unique()))
    # print train.race.value_counts()

