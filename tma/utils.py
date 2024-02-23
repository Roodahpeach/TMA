import pandas as pd

class LogLog:
    def __init__(self, filepath: str, OpenTextMode="w") -> None:
        self.filepath = filepath
        with open(self.filepath, OpenTextMode):
            pass
    def add(self, string):
        with open(self.filepath, "a") as f:
            f.write(f"{str(string)}")
            f.write("\n")

class SmartDict:
    def __init__(self) -> None:
        self.counter = {}
        self.is_array = False
    def add(self, index, value):
        if not (index in self.counter.keys()):
            self.counter[index] = 0
        self.counter[index] += value
    def array_add(self, x, y, value):
        index = (x, y)
        if not (index in self.counter.keys()):
            self.counter[index] = 0
        self.counter[index] += value
        self.is_array = True
    def append(self, key, value):
        if not (key in self.counter.keys()):
            self.counter[key] = []
        self.counter[key].append(value)
    def to_file(self, save_path):
        if self.is_array:
            with open(save_path, "w") as f:
                f.write("x,y, value\n")
                for key in self.counter.keys():
                    f.write(
                        f"{str(key[0])},{str(key[1])},{str(self.counter[key])}"
                    )
                    f.write("\n")
        else:
            with open(save_path, "w") as f:
                f.write("key, value\n")
                for key in self.counter.keys():
                    f.write(f"{str(key)},{str(self.counter[key])}")
                    f.write("\n")
    def to_list_of_dict(self):
        result = []
        if self.is_array:
            for key in self.counter.keys():
                result.append(
                    {"x": key[0], "y": key[1], "value": self.counter[key]}
                )
        else:
            for key in self.counter.keys():
                result.append({"key": key, "value": self.counter[key]})
        return result
    def to_dict(self):
        return self.counter
    def to_dataframe(self):
        return pd.DataFrame(self.to_list_of_dict())