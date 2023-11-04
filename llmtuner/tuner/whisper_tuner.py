from llmtuner.models.whisper_model import WhisperModel
from llmtuner.llmtrainer.whisper_trainer import WhisperModelTrainer
from llmtuner.Inference.inference_pipeline import WhisperEval
import gradio as gr

class Tuner:
    def __init__(self, model, dataset, processor, training_args_dict=None):
        self.model = model
        self.dataset = dataset
        self.processor = processor
        self.training_args_dict = training_args_dict
        self.trained = False

    def fit(self):
        # Load and initialize the model
        model = self.model.load_model()
        
        # Preprocess the dataset
        processed_dataset = self.processor.preprocess_data(self.dataset)

        # Initialize and set up the trainer
        trainer = WhisperModelTrainer(model, processed_dataset, self.processor, self.model.use_peft)
        if self.training_args_dict:
            trainer.setup_trainer(**self.training_args_dict)

        # Start the training process
        trainer.start_training()

        self.model = model
        self.trained = True
        return model

    def inference(self, audio_file):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before performing inference.")
        
        transcription_pipeline = WhisperEval(self.model, 
                                             self.model.language_abbr, 
                                             self.model.task, 
                                             self.model.use_peft)
        return transcription_pipeline.transcribe(audio_file)

    def wer_eval(self, split):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before evaluating WER.")

        transcription_pipeline = WhisperEval(self.model, 
                                             self.model.language_abbr, 
                                             self.model.task, 
                                             self.model.use_peft)
        return transcription_pipeline.evaluate_dataset(self.dataset[split])

    def launch_ui(self):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before launching the UI.")

        pipe = WhisperEval(self.model, 
                           self.model.language_abbr, 
                           self.model.task, 
                           self.model.use_peft)

        iface = gr.Interface(fn=pipe.transcribe,
                             inputs=gr.Audio(source="microphone", type="filepath"),
                             outputs="text",
                             title="Whisper Demo",
                             description="Realtime demo for speech recognition using a fine-tuned Whisper model.")
        iface.launch()

    def save(self, filepath):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before saving.")
        
        if hasattr(self.model, 'save'):
            self.model.save(filepath)
        else:
            raise AttributeError("The model instance does not have a 'save' method.")