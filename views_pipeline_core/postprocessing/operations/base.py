from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PostProcessOperation(ABC):
    """
    Abstract base class for all postprocessing operations.
    """
    
    @abstractmethod
    def process(self, data, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        """Allow steps to be called like functions."""
        op_name = self.__class__.__name__
        logger.info(f"Starting postprocessing operation: {op_name}")
        return self.process(data, **kwargs)