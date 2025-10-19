from decimal import Decimal
from dataclasses import dataclass, field
import json
import dataclasses

@dataclass
class PreProcessed:
    train_index_start:int
    val_index_start:int
    test_index_start:int
    train_index_end:int
    val_index_end:int
    test_index_end:int
    train_set_length:int
    val_set_length:int
    test_set_length:int
    file:str

    def to_dict(self):
        return {
            'train_index_start':self.train_index_start,
            'val_index_start':self.val_index_start,
            'test_index_start':self.test_index_start,
            'train_index_end':self.train_index_end,
            'val_index_end':self.val_index_end,
            'test_index_end':self.test_index_end,
            'train_set_length':self.train_set_length,
            'val_set_length':self.val_set_length,
            'test_set_length':self.test_set_length,
            'file':self.file
        }
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


class PreProcessedEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, Decimal):
                return str(o)
            return super().default(o)
