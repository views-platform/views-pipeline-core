from typing import Dict, List, Any
import pandas as pd
import logging
from .registry import POSTPROCESSOR_CATALOG

logger = logging.getLogger(__name__)


class PostprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.postprocessor_config = config.get("postprocessor", {})
        self.pipeline = []
        
        for name, post_config in self.postprocessor_config.items():
            if name in POSTPROCESSOR_CATALOG:
                processor_class = POSTPROCESSOR_CATALOG[name]
                self.pipeline.append(processor_class(post_config, self.config))
            else:
                logger.warning(f"Requested postprocessor '{name}' not found in registry.")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.pipeline:
            return df
            
        current_df = df.copy()
        for processor in self.pipeline:
            logger.info(f"Applying postprocessor: {processor.__class__.__name__}")
            current_df = processor.apply(current_df)
            
        return current_df