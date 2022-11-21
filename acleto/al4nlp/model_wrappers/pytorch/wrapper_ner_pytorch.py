import logging

from ..pytorch.wrapper_encoder_pytorch import PytorchEncoderWrapper
from ..transformers.wrapper_ner import WrapperNer
from ...models import PYTORCH_INIT_MODELS_DICT

log = logging.getLogger()


class PytorchNerWrapper(PytorchEncoderWrapper, WrapperNer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model_class(self):
        try:
            return PYTORCH_INIT_MODELS_DICT[self.model_config.checkpoint]["model_class"]
        except:
            model_class, _ = self.get_model_from_path()
            return model_class
