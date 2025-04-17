from typing import Dict, Type
from views_pipeline_core.postprocessing.operations.base import PostProcessOperation


class OperationRegistry:
    """
    Manages registration and retrieval of processing operations.
    """
    _instance = None
    _operations: Dict[str, Type[PostProcessOperation]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str) -> callable:
        """
        Decorator to register a operation class.
        """
        def decorator(operation_class: Type[PostProcessOperation]):
            if not issubclass(operation_class, PostProcessOperation):
                raise TypeError(f"{operation_class.__name__} must inherit from PostProcessOperation")
            cls._operations[name] = operation_class
            return operation_class
        return decorator


    @classmethod
    def get(cls, name: str, **kwargs) -> PostProcessOperation:
        """
        Instantiate a operation by name with optional kwargs.
        Args:
            name (str): Name of the operation to instantiate.
            kwargs: Additional parameters for the operation constructor.
        Returns:
            PostProcessOperation: An instance of the requested operation.
        """
        if name not in cls._operations:
            raise KeyError(f"Operation '{name}' not found. Available: {list(cls._operations.keys())}")
        return cls._operations[name](**kwargs)

    @classmethod
    def list_operations(cls) -> Dict[str, str]:
        """
        Return registered operations.
        """
        return [name for name in cls._operations.keys()]

