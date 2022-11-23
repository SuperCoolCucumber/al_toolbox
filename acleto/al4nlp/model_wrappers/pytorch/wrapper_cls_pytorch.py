import logging

from .wrapper_encoder_pytorch import PytorchEncoderWrapper
from ..transformers.wrapper_cls import WrapperCls
from ...models import PYTORCH_INIT_MODELS_DICT

log = logging.getLogger()


class PytorchClsWrapper(PytorchEncoderWrapper, WrapperCls):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model_class(self):
        try:
            return PYTORCH_INIT_MODELS_DICT[self.model_config.checkpoint]["model_class"]
        except:
            model_class, _ = self.get_model_from_path()
            return model_class
