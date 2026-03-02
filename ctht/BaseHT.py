from abc import abstractmethod, ABC


class BaseHT(ABC):
    def __init__(self, img_width):
        self.img_width = img_width

    @abstractmethod
    def fit(self, pts, use_max_n_lines=None):
        pass
