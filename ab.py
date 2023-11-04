from llmtuner import WhisperModel
from llmtuner import PeftWhisperModel

# model_ = WhisperModel("openai/whisper-small")
# model = model_.load()

peftmodel_ = PeftWhisperModel("openai/whisper-small")
peftmodel = peftmodel_.load()