from llmtuner.models.whisper_model import WhisperModel
from llmtuner.trainer.whisper_trainer import WhisperModelTrainer
from llmtuner.Inference.inference_pipeline import WhisperEval
from llmtuner.dataset.whisper_data import AudioDatasetProcessor
import gradio as gr

class Tuner:
    def __init__(self, model, dataset, training_args_dict=None):
        
        self.model_ = model
        self.model  = None
        self.dataset = dataset
        self.processor = None
        self.training_args_dict = training_args_dict
        self.trained = False
        self.trained_model_path = None

    def fit(self):
        # Load and initialize the model

        model = self.model_.load_model()
        self.processor = AudioDatasetProcessor(self.model_.model_name_or_path, self.model_.language, self.model_.task)
        # Preprocess the dataset
        processed_dataset, feat_processor = self.processor.preprocess_data(self.dataset.dataset)
        # Initialize and set up the trainer
        trainer = WhisperModelTrainer(model, processed_dataset, feat_processor)
        if self.training_args_dict:
            trainer.setup_trainer(self.training_args_dict)
        else:
            trainer.setup_trainer()

        # Start the training process
        trainer.start_training()

        self.model = model
        self.trained = True
        self.trained_model_path = trainer.model_saved_path
        return model

    def inference(self, audio_file, language_abbr = 'hi'):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before performing inference.")
        
        transcription_pipeline = WhisperEval(self.trained_model_path, 
                                             language_abbr, 
                                             self.model_.task, 
                                             self.model_.use_peft)
        return transcription_pipeline.transcribe(audio_file)

    def wer_eval(self, split, language_abbr = 'hi'):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before evaluating WER.")

        transcription_pipeline = WhisperEval(self.trained_model_path, 
                                             language_abbr, 
                                             self.model_.task, 
                                             self.model_.use_peft)
        return transcription_pipeline.evaluate_dataset(self.dataset[split])

    def launch_ui(self, language_abbr = 'hi', share=True):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before launching the UI.")

        pipe = WhisperEval(self.trained_model_path, 
                          language_abbr, 
                           self.model_.task, 
                           self.model_.use_peft)

        iface = gr.Interface(fn=pipe.transcribe,
                             inputs=gr.Audio(source="microphone", type="filepath"),
                             outputs="text",
                             title="Whisper Speech Recognition",
                             description="Fine-Tuned Whisper Model for Realtime speech recognition")
        iface.launch(share=share)

    def save(self, filepath):
        if not self.trained:
            raise Exception("Model has not been trained yet. Call `fit` before saving.")
        
        if hasattr(self.model, 'save'):
            self.model.save(filepath)
        else:
            raise AttributeError("The model instance does not have a 'save' method.")