import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(filename):
    df = pd.read_csv(filename).set_index('ID')
    df['Код_группы'] = df['Код_группы'].astype('category')
    df['Код1'] = df['Код_группы'].apply(lambda x: str(x)[:2]).astype('category')
    df['Код2'] = df['Код_группы'].apply(lambda x: str(x)[2:4]).astype('category')
    df['Код3'] = df['Код_группы'].apply(lambda x: str(x)[2:]).astype('category')

    df['Муж'] = (df['Пол'].str.lower()=='муж').astype(int)
    df.drop(['Пол'], axis=1, inplace=True)
    
    df['Основания'] = df['Основания'].str.lower().astype('category')

    # df['Изучал_Англ'] = df['Изучаемый_Язык'].fillna('').str.lower().str.contains('англ').astype('int')
    df.drop(['Изучаемый_Язык'], axis=1, inplace=True)

    df['Год_Рождения'] = pd.to_datetime(df['Дата_Рождения']).dt.year
    df.drop(['Дата_Рождения'], axis=1, inplace=True)

    df['Возраст_Поступления'] = df['Год_Поступления'] - df['Год_Рождения']
    df['Перерыв'] = (df['Год_Поступления'] - df['Год_Окончания_УЗ']).fillna(0).astype(int)
    df.drop(['Год_Окончания_УЗ'], axis=1, inplace=True)

    # df['Общежитие'] = df['Общежитие'].fillna(0).astype(bool)

    df['ПолнаяСемья'] = ((df['Наличие_Матери']>0)&(df['Наличие_Отца']>0)).astype(bool)
    df.drop(['Наличие_Матери', 'Наличие_Отца'], axis=1, inplace=True)
    # df['Наличие_Матери'] = df['Наличие_Матери'].astype(int)
    # df['Наличие_Отца'] = df['Наличие_Отца'].astype(int)

    df['КодФакультета'] = df['КодФакультета'].astype(int).astype('category')
    
    df['СрБаллАттестата'] = np.where(df['СрБаллАттестата']<=5, df['СрБаллАттестата']*100/5, df['СрБаллАттестата'])

    schools = ['сош', 'мбоу', 'кгбоу', 'мкоу', 'кгу', 'сш', 'соу', 'моу', 'мсош', 'школа', 'лицей', 'гимназия', 'академия', 'корпус', 'школы']
    colleges = ['пу', 'нпо', 'спо', 'впо', 'спту', 'училище', 'техникум', 'колледж', 'коледж', 'коллежд', 'колледжа', 'профлицей']
    universities = ['фгбоу', 'фбгоу', 'университет', 'институт', 'универститет', 'универсиет', 'консерватория', 'алтгу', 'взфэи', 'бюи', 'уриверситет']

    places = df['Уч_Заведение'].fillna('')
    def f(df, l):
        seps = ['-', '"', '(', ')', '№', '»']
        s = df.lower()
        for sep in seps:
            s = s.replace(sep, ' ')
        df = len(set(s.split(' '))&set(l))
        return df
    is_school = places.apply(lambda x: f(x, schools)>0)
    is_college = places.apply(lambda x: f(x, colleges)>0)
    is_university = places.apply(lambda x: f(x, universities)>0)
    # print(df.loc[is_university])
    df['Учеба'] = 'н'
    df.loc[is_university, 'Учеба'] = 'у'
    df.loc[is_college, 'Учеба'] = 'к'
    df.loc[is_school, 'Учеба'] = 'ш'
    df['Учеба'] = df['Учеба'].astype('category')
    df.drop(['Уч_Заведение'], axis=1, inplace=True)

    # school_loc = df['Где_Находится_УЗ'].fillna('')
    # barnaul_loc = school_loc.str.contains('барнаул', case=False)
    # alt_loc = school_loc.str.contains('алтайский', case=False)
    # df['МестоУчебы'] = 'д' # другое
    # df.loc[alt_loc, 'МестоУчебы'] = 'к' # алтайский край
    # df.loc[barnaul_loc, 'МестоУчебы'] = 'б' # барнаул
    # df['МестоУчебы'] = df['МестоУчебы'].astype('category')
    df.drop(['Где_Находится_УЗ'], axis=1, inplace=True)

    loc_list1 = {
        'росс' : 'р',
        'казах' : 'снг',
        'украин' : 'снг',
        'к[иы]ргиз' : 'снг',
        'таджик' : 'снг',
        'арм' : 'снг',
        'туркм' : 'снг',
        'узбек' : 'снг',
    }
    loc_list2 = {
        'алтайский' : 'алт',
    }
    loc_list3 = {
        'барнаул' : 'брн',
        'бийск' : 'бск',
        'новоалтайск' : 'нва',
    }
    df['МестоЖит'] = 'д' # другое

    def fill_location(input_col, output_col, location_list):
        loc = df[input_col].fillna('')
        for key in location_list:
            true_loc = loc.str.contains(key, case=False, regex=True)
            df.loc[true_loc, output_col] = location_list[key]
        df.drop([input_col], axis=1, inplace=True)
        return

    fill_location('Страна_ПП', 'МестоЖит', loc_list1)
    fill_location('Регион_ПП', 'МестоЖит', loc_list2)
    fill_location('Город_ПП', 'МестоЖит', loc_list3)
    df['МестоЖит'] = df['МестоЖит'].astype('category')

    df.drop(['Пособие', 'Опекунство', 'Общежитие',
        'Страна_Родители', 'Село', 'Иностранец'], axis=1, inplace=True)
    return df

def train_default_catboost(X, y, X_test, params={}, savepath='catboost_default.csv'):
    """0.7919 public testboard"""
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    print(f'Train len: {X_train.shape[0]}, val len: {X_val.shape[0]}, test len: {X_test.shape[0]}')

    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import RandomizedSearchCV

    cat_features=np.arange(len(X_train.columns))[X_train.dtypes == 'category']
    data = Pool(X_train, label=y_train, cat_features=cat_features)

    clf = CatBoostClassifier(verbose=False, **params)

    clf.fit(data)

    print(clf.get_all_params())
    from sklearn.metrics import f1_score

    pred0 = clf.predict(X_train, prediction_type='Class')
    pred = clf.predict(X_val, prediction_type='Class')

    f1_train = f1_score(y_train, pred0, average='macro', zero_division = 0)
    f1_val = f1_score(y_val, pred, average='macro', zero_division = 0)
    f1_val_classes = f1_score(y_val, pred, average=None)
    
    print('F1 train:', f1_train, 'F1 test:', f1_val)
    print('F1 test classes:', f1_val_classes)

    print(dict(zip(X_train.columns, clf.feature_importances_)))
    
    if savepath is None:
        return clf

    clf = CatBoostClassifier(verbose=False, **params)
    clf.fit(Pool(X, label=y, cat_features=cat_features))
    
    print('Final clf params:')
    print(clf.get_all_params())

    pred_test = clf.predict(X_test, prediction_type='Class').ravel()
    pd.Series(pred_test, index=X_test.index, name='Статус').replace({0: -1, 1: 3, 2: 4}).to_csv(savepath)
    return clf

def cross_validation(X, y, params, fold_count: int = 3):
    from catboost import cv, Pool
    cat_features = np.arange(len(X.columns))[X.dtypes == 'category']
    
    cv_data = cv(
        params=params,
        pool=Pool(X, label=y, cat_features=cat_features),
        fold_count=fold_count,
        shuffle=True,
        partition_random_seed=0,
        plot=False,
        stratified=True, 
        verbose=False,
        return_models=False,
    )    
    return cv_data

def plot_cv_learning_curve(cv_data, param, fold_count, title=None):
    import matplotlib.pyplot as plt
    def plot_line(x, y, yerr, ax, label):
        label = f'{label} ({y.iloc[-1]:.3f}±{yerr.iloc[-1]:.3f})'
        ax.errorbar(x, y, label=label)
        ax.fill_between(x, y-yerr, y+yerr, alpha=0.3)
    
    fig, ax = plt.subplots(1, 1, dpi=120)
    plot_line(
        cv_data.iterations,
        cv_data[f'train-{param}-mean'], 
        cv_data[f'train-{param}-std']/np.sqrt(fold_count), 
        ax, 'train')

    plot_line(
        cv_data.iterations,
        cv_data[f'test-{param}-mean'], 
        cv_data[f'test-{param}-std']/np.sqrt(fold_count), 
        ax, 'val')

    ax.set(xlabel='Iteration', ylabel='F1 Macro', xlim=(0, None), title='' if title is None else title)
    ax.legend();
    return

def gridsearch(X, y, params, fold_count=3, savefolder='./results/images'):
    from sklearn.model_selection import ParameterGrid
    
    pgrid = ParameterGrid(params)
    plt.ioff()
    for pars in pgrid:
        print(pars)
        cv_data = cross_validation(X, y, pars, fold_count)
        
        metric = pars.pop('eval_metric')
        title = ';'.join(map(lambda x: f'{x[0]}:{x[1]}', pars.items()))
        
        plot_cv_learning_curve(cv_data, metric, fold_count, title)
        plt.savefig(f'{savefolder}/cv_{title}.png')
        print(f'cv_{title}.png', 'is done')
    # plt.ion()
    return 
    