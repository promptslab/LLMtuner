from llmtuner.models.whisper_model import WhisperModel
from llmtuner.models.peft_whisper import PeftWhisperModel

class Model:
    def __init__(self, model_name_or_path, language="Hindi", 
                 language_abbr = 'hi',
                 task="transcribe", use_peft=False, 
                 peft_config=None, quantization_config=None):
        
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.language_abbr = language_abbr
        self.task = task
        self.use_peft = use_peft
        self.peft_config = peft_config
        self.quantization_config = quantization_config
        self.model_instance = None

    def load_model(self):
        if self.use_peft:
            self.model_instance = PeftWhisperModel(
                model_name_or_path=self.model_name_or_path,
                language=self.language,
                task=self.task,
                peft_config=self.peft_config,
                quantization_config=self.quantization_config
            )
        else:
            self.model_instance = WhisperModel(
                model_name_or_path=self.model_name_or_path,
                language=self.language,
                task=self.task
            )

        return self.model_instance.load()

    def save_model(self, save_path):
        if self.model_instance is not None:
            self.model_instance.save(save_path)
        else:
            raise ValueError("Model is not loaded. Please load a model before trying to save.")