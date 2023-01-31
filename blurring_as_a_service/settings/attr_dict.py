from typing import Dict


class AttrDict(dict):
    """Dictionary which also allows access to keys as an attribute.

    Parameters
    ----------
    d : dict, optional
         if provided, initialize the `AttrDict` with the data from `d` using
         `_data_from_dict()`. Else, initialize an empty `AttrDict`.
    """

    def __init__(self, d: Dict = None):
        super().__init__()
        if d:
            ad = self._data_from_dict(d)
            for k, v in ad.items():
                self[k] = v

    @classmethod
    def _data_from_dict(cls, d: Dict):
        """Recursively converts a dictionary to an `AttrDict`.

        It follows this algorithm:
        1. loop through all key/value pairs
        2. if value is a dict: create an `AttrDict` from it and add it to the output dict
           under `key`
        3. else: add the key/value pair to the output dict
        """
        data = {}
        for k, v in d.items():
            if isinstance(v, dict):
                data[k] = cls(v)
            else:
                data[k] = cls.process_value(k, v)
        return data

    @classmethod
    def process_value(cls, k, v):
        if isinstance(v, list):
            return [cls.process_value(f"{k}.item", item) for item in v]
        else:
            return v

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value
