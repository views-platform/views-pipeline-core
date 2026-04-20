from abc import ABC, abstractmethod
import pandas as pd

class BasePostprocessor(ABC):
    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _validate(self, smoothed_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass