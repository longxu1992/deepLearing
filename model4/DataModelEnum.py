from enum import Enum


class SqlDetail:
    SQL1 = ''' SELECT 
 
    '''
    SQL2 = ''' SELECT 

    '''

    SQLT1 = ''' SELECT 

    '''
    SQLT2 = ''' SELECT 
    '''


class DataModelEnum(Enum):
    M8D1 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQL1)
    M8T1 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQLT1)
    M8D2 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQL2)
    M8T2 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQLT2)
    M7D1 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQL1)
    M7T1 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQLT1)
    M7D2 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQL2)
    M7T2 = ("../东方财富web.db", "history_trend_num_final", 229, SqlDetail.SQLT2)

    def __init__(self, db_path, table_name, input_size, sql):
        self.db_path = db_path
        self.table_name = table_name
        self.input_size = input_size
        self.sql = sql
