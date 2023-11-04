from transformers import WhisperForConditionalGeneration
from .base_model import BaseModel


class WhisperModel(BaseModel):
    def __init__(self, model_name_or_path, language="Hindi", task="transcribe"):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.task = task
        self.model = None

    def load(self):
        if not self.model:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name_or_path)
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
        return self.model
    
    def save(self, save_path, *args, **kwargs):
        self.model.save_pretrained(save_path, from_pt=True)





    
