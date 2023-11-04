import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from .base_dataset import BaseDatasetProcessor

class AudioDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, model):
        
        super().__init__()
        self.model_ = model
        self.language = self.model_.language
        self.language_abbr = self.model_.language_abbr
        self.task = self.model_.task
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_.model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_, language=self.language, task=self.task
        )
        self.processor = WhisperProcessor.from_pretrained(
            self.model_, language=self.language, task=self.task
        )

    def preprocess_data(self, hf_audio_object):
        # Cast the column to the desired format
        audio_data = hf_audio_object.cast_column("audio", Audio(sampling_rate=16000))
        # Apply preprocessing
        audio_data = audio_data.map(
            self._prepare_dataset,
            remove_columns=hf_audio_object.column_names["train"],
            num_proc=2,
        )
        
        return audio_data

    def _prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["sentence"] = self.tokenizer._normalize(batch["sentence"])
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        
        return batch