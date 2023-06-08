import sqlite3
import numpy as np


class DataReaderPrd:
    def __init__(self, data_enum, week_number, day_of_week):
        self.conn = sqlite3.connect(data_enum.db_path)
        self.cursor = self.conn.cursor()
        self.sql = data_enum.sql
        self.model_db = data_enum.name
        self.week_number = week_number
        self.day_of_week = day_of_week

    def get_batch(self):
        # self.cursor.execute(f'SELECT * FROM {self.table_name} LIMIT {batch_size} OFFSET {self.offset}')
        self.cursor.execute(f'''
                {self.sql} 
                and  rq_week0 = {self.week_number} AND rq_day0 = {self.day_of_week}
        ''')
        result = self.cursor.fetchall()
        if len(result) == 0:
            return None
        result = np.array(result, dtype=np.float32)
        return result
