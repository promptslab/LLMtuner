from llmtuner.trainer.base_trainer import BaseTrainer
from llmtuner.Inference.metrices import WERMetrics
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class WhisperModelTrainer(BaseTrainer):
    def __init__(self, model, 
                 processed_data,
                 processor,
                 output_dir = None):
        
        super().__init__(output_dir, task)
        self.model      = model
        self.processed_data = processed_data
        self.processor = processor
        self.trainer = None
        self.wer_metrics = WERMetrics(self.processor.tokenizer)

    def setup_trainer(self, training_args_dict=None):
        # Define default arguments for training
        default_args = {
            'output_dir': self.output_dir,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'learning_rate': 1e-3,
            'warmup_steps': 10,
            'max_steps': 10,
            'gradient_checkpointing': True,
            'fp16': True,
            'evaluation_strategy': "steps",
            'per_device_eval_batch_size': 8,
            'generation_max_length': 225,
            'save_steps': 5,
            'eval_steps': 5,
            'logging_steps': 5,
            'report_to': ["wandb"],
            'load_best_model_at_end': True,
            'push_to_hub': True
        }

        combined_args = {**default_args, **training_args_dict} if training_args_dict else default_args
        training_args = Seq2SeqTrainingArguments(**combined_args)

        # Customize the training arguments based on the type of model
        if self.model.is_peft_applied:
            # Settings specific to PEFT model
            training_args.label_names = ["labels"]
            training_args.remove_unused_columns = False
            compute_metrics = None
        else:
            # Settings specific to Whisper model
            training_args.predict_with_generate = True
            training_args.metric_for_best_model = "wer"
            training_args.greater_is_better = False
            compute_metrics = self.wer_metrics.compute_metrics

        # Initialize Seq2SeqTrainer with the provided setup
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.processed_data["train"],
            eval_dataset=self.processed_data["test"],
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            tokenizer=self.processor.feature_extractor,
            compute_metrics=compute_metrics
        )

        # Save the processor
        self.processor.processor.save_pretrained(training_args.output_dir)

    def start_training(self):
        # Implement the specific training loop for Whisper or PEFT models
        self.trainer.train()

    def push_to_hub(self, hub_push_kwargs=None):
        hub_push_kwargs_default = {
            "language": 'hi',
            "model_name": self.output_dir,
            "finetuned_from": self.model_name,
            "tasks": "automatic-speech-recognition",
            "tags": self.task,
        }
        hub_combined_args = {**hub_push_kwargs_default, **hub_push_kwargs} if hub_push_kwargs else hub_push_kwargs_default

        # Call the base class's push_to_hub with combined arguments
        self.trainer.push_to_hub(**hub_combined_args)