from typing import Type
from .base import BasePostprocessor

POSTPROCESSOR_CATALOG = {}

def register_postprocessor(name: str, processor_cls: Type[BasePostprocessor]):
    """Allows downstream packages to register their custom postprocessors."""
    POSTPROCESSOR_CATALOG[name] = processor_cls