from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from .base_model import BaseModel
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class PeftWhisperModel(BaseModel):
    def __init__(self, model_name_or_path, peft_config=None, language="Hindi", task="transcribe", quantization_config=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.task = task
        self.peft_config = peft_config
        self.quantization_config = self._get_default_quantization_config()
        self.model = None

        if quantization_config:
            self.quantization_config.update(quantization_config)

    def _get_default_quantization_config(self):
        # Default configuration with 8-bit precision
        return BitsAndBytesConfig(load_in_8bit=True)

    def _apply_peft_to_model(self, model):
        
        model = prepare_model_for_kbit_training(model)
        # Here, we'll apply the forward hook to make sure gradients are computed
        # for the modified layers. This function will be used as the hook.
        model.model.encoder.conv1.register_forward_hook(self.make_inputs_require_grad)
        
        # Use the provided PEFT config or create a default one
        peft_config = self.peft_config if self.peft_config else LoraConfig(
            r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        
        # Apply PEFT configuration
        peft_model = get_peft_model(model, peft_config)
        print(peft_model.print_trainable_parameters())
        return peft_model

    def make_inputs_require_grad(self, module, input, output):
        output.requires_grad_(True)

    
    def load(self):
        if not self.model:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                quantization_config=self.quantization_config,
                device_map="auto"
            )

            # Apply PEFT configuration
            self.model = self._apply_peft_to_model(self.model)
        return self.model

    def save(self, save_path, *args, **kwargs):
        if self.model:
            self.model.save_pretrained(save_path)
        else:
            raise ValueError("Model is not loaded. Please call 'load' before saving.")