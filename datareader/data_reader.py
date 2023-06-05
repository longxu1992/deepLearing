import sqlite3
import numpy as np


class DataReader:
    def __init__(self, data_enum):
        self.conn = sqlite3.connect(data_enum.db_path)
        self.cursor = self.conn.cursor()
        self.sql = data_enum.sql
        self.model_db = data_enum.name
        self.offset = 0

    def get_batch(self, batch_size):
        # self.cursor.execute(f'SELECT * FROM {self.table_name} LIMIT {batch_size} OFFSET {self.offset}')
        self.cursor.execute(f'''
                {self.sql}
            LIMIT {batch_size} OFFSET {self.offset}
        ''')
        result = self.cursor.fetchall()
        if len(result) == 0:
            return None, None
        result = np.array(result, dtype=np.float32)
        X = result[:, :-1]
        y = result[:, -1:]
        y = np.where(y > 0, 1, 0)
        self.offset += batch_size
        return X, y
