import pandas as pd

_type = ['customer', 'product', 'area', 'sale', 'date']

class mainColunm:
    _type = ['customer', 'product', 'area']
    def __init__(self, col = pd.Series, segment = pd.Series, segment_chose = [], type = str):
        if type not in mainColunm._type:
            raise TypeError(f'mainColumn type must be in {mainColunm._type}')
        self.type = type
        self.main = col
        self.segment = segment
        self.chose = segment_chose
        if segment is not None:
            input = pd.merge(self.main, segment, how = 'inner', left_index = True, right_index = True)
            if self.chose == []:
                self.data = input
            else:
                self.data = input[input[segment.name].isin(segment_chose)]
        else:
            self.data = pd.DataFrame(self.main)
            self.data['segment'] = 'no segment'
        self.data.columns = [f'{self.type}', f'{self.type}_segment']

    def __repr__(self):
        return f'{self.type} col by {self.main.name} - chose {self.chose} in {self.segment.name}'
    
    def __len__(self):
        return len(self.data)

class Sale:
    def __init__(self, data = pd.Series):
        if data.dtype not in ['float', 'int']:
                raise TypeError(f'sales columns need dtype float or int')
        self.main = data
        self.data = pd.DataFrame(data)
        self.data.columns = ['sale']
        self.type = 'sale'

    def __repr__(self):
        return f'{self.type} col by {self.main.name}'
    
    def __len__(self):
        return len(self.data)

class Date:
    _column = ['year', 'month', 'day', 'y_m']
    def __init__(self, data = pd.Series, date_format = '%m-%d-%Y',
                 year_chose = [], month_chose = [], day_chose = []):
        try:
            data = pd.to_datetime(data, format = date_format)
        except:
            raise ValueError(f'Can not parse date with format {date_format}')
        self.type = 'date'
        for i in [year_chose, month_chose, day_chose]:
            if not isinstance(i, list):
                raise TypeError(f'{i} must be a list')
        if year_chose == []:
            self.year = sorted(data.dt.year.unique().tolist())
        else:
            self.year = year_chose
        if month_chose == []:
            self.month = sorted(data.dt.month.unique().tolist())
        else:
            self.month = month_chose
        if day_chose == []:
            self.day = sorted(data.dt.day.unique().tolist())
        else:
            self.day = day_chose
        data = data.loc[data.dt.year.isin(self.year)&
                             data.dt.month.isin(self.month)&
                             data.dt.day.isin(self.day)]
        self.data = pd.DataFrame({'date': data,
                                  'year': data.dt.year,
                                  'month': data.dt.month,
                                  'day': data.dt.day,
                                  'y_m': data.dt.strftime('%y_%m')})
        self.index = self.data.index.tolist()
        self.name = data.name
        
    def __repr__(self):
        return f'Datecol:\nby {self.name}\nyear{self.year}\nmonth{self.month}\nday{self.day}'
    
    def __len__(self):
        return len(self.data)
