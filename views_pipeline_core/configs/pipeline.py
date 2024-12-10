import logging
logger = logging.getLogger(__name__)

class PipelineConfig:
    def __init__(self):
        self._dataframe_format = '.pkl'
        self._model_format = '.pkl'

    @property
    def dataframe_format(self) -> str:
        logger.debug(f"Dataframe format: {self._dataframe_format}")
        return self._dataframe_format

    @dataframe_format.setter
    def dataframe_format(self, format: str):
        logger.debug(f"Setting dataframe format: {format}")
        self._dataframe_format = format

    # @property
    # def model_format(self) -> str:
    #     logger.debug(f"Model format: {self._model_format}")
    #     return self._model_format

    # @model_format.setter
    # def model_format(self, format: str):
    #     logger.debug(f"Setting model format: {format}")
    #     self._model_format = format
