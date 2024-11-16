from abc import abstractmethod, ABC


class App(ABC):
    @abstractmethod
    def run(self):
        pass
