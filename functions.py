import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(filename):
    df = pd.read_csv(filename).set_index('ID')
    df['Код_группы'] = df['Код_группы'].astype('category')
    df['Код1'] = df['Код_группы'].apply(lambda x: str(x)[:2]).astype('category')
    df['Код2'] = df['Код_группы'].apply(lambda x: str(x)[2:4]).astype('category')
    df['Код3'] = df['Код_группы'].apply(lambda x: str(x)[2:]).astype('category')

    df['Муж'] = (df['Пол'].str.lower()=='муж').astype(bool)
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

    schools = ['сош', 'мбоу', 'мкоу', 'кгу', 'сш', 'соу', 'моу', 'мсош', 'школа', 'лицей', 'гимназия', 'академия', 'корпус', 'школы']
    colleges = ['кгбоу', 'пу', 'нпо', 'спо', 'впо', 'спту', 'училище', 'техникум', 'колледж', 'коледж', 'коллежд', 'колледжа', 'профлицей', 'профессионально']
    universities = ['фгбоу', 'фбгоу', 'университет', 'институт', 'универститет', 'универсиет', 'консерватория', 'алтгу', 'взфэи', 'бюи', 'уриверситет']
    asu_uni = [r'алт\w+\s+гос\w+\sуни\w+', 'алтгу']

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
    for asu in asu_uni:
        true_place = places.str.contains(asu, regex=True, case=False)
        df.loc[true_place, 'Учеба'] = 'алтгу'
    
    
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
        # 'новосиб' : 'нсо',
        # 'кемер': 'кем',
        r'алтай\b': 'ралт',
    }
    loc_list3 = {
        'барнаул' : 'брн',
        'бийск' : 'бск',
        'новоалтайск' : 'нва',
        'заринск': 'зрн',
        'рубцовск': 'рбц',
        'славгород': 'слв',
        'яровое': 'слв',
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
    df['МестоЖит'] = np.where((df['Село']>0)&(df['МестоЖит']=='алт'), 'алтс', df['МестоЖит'])
    
    df['МестоЖит'] = df['МестоЖит'].astype('category')

    df.drop(['Пособие', 'Опекунство', 'Общежитие',
        'Страна_Родители', 'Село', 'Иностранец'], axis=1, inplace=True)
    
    if 'Статус' in df:
        # только для train
        df.drop_duplicates(inplace=True)
    
    return df

def train_default_catboost(X, y, X_test, savepath='catboost_default.csv'):
    """0.7919 public testboard"""
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print(f'Train len: {X_train.shape[0]}, val len: {X_val.shape[0]}, test len: {X_test.shape[0]}')

    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import RandomizedSearchCV

    cat_features=np.arange(len(X_train.columns))[X_train.dtypes == 'category']
    data = Pool(X_train, label=y_train, cat_features=cat_features)

    clf = CatBoostClassifier(verbose=False)

    clf.fit(data)

    from sklearn.metrics import f1_score

    pred0 = clf.predict(X_train, prediction_type='Class')
    pred = clf.predict(X_val, prediction_type='Class')

    f1_train = f1_score(y_train, pred0, average='macro', zero_division = 0)
    f1_val = f1_score(y_val, pred, average='macro', zero_division = 0)

    print('F1 train:', f1_train, 'F1 test:', f1_val)

    print(dict(zip(X_train.columns, clf.feature_importances_)))

    clf = CatBoostClassifier(verbose=False)
    clf.fit(Pool(X, label=y, cat_features=cat_features))

    pred_test = clf.predict(X_test, prediction_type='Class').ravel()
    pd.Series(pred_test, index=X_test.index, name='Статус').replace({0: -1, 1: 3, 2: 4}).to_csv(savepath)
    return clf