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

class AudioDataProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        whisper_model="openai/whisper-small",
        language="Hindi",
        task="transcribe",
    ):
        super().__init__()
        self.whisper_model = whisper_model
        self.language = language
        self.task = task
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisper_model)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.whisper_model, language=self.language, task=self.task
        )
        self.processor = WhisperProcessor.from_pretrained(
            self.whisper_model, language=self.language, task=self.task
        )
    
    def load_data(
        self, train_dir="train_data/", test_dir="test_data/", dummy=False, type_="full"
    ):
        if dummy:
            # Load dummy data for testing purposes
            self.data = self.load_dummy_data(type_)
        else:
            # Load actual data from directories
            hf_dataset = DatasetDict()
            hf_dataset["train"] = load_dataset(
                "audiofolder", data_dir=train_dir, split="train"
            )
            hf_dataset["test"] = load_dataset(
                "audiofolder", data_dir=test_dir, split="train"
            )
            self.data = hf_dataset
        return self.data

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

    def load_dummy_data(self, type_="full"):
        """
        Load dummy data from the `mozilla-foundation/common_voice_11_0` dataset.

        Returns:
        DatasetDict: A dictionary containing the train and test datasets with selected columns removed.
        """
        common_voice = DatasetDict()

        if type_ == "full":
            common_voice["train"] = load_dataset(
                "mozilla-foundation/common_voice_11_0", "hi", split="train+validation"
            )
            common_voice["test"] = load_dataset(
                "mozilla-foundation/common_voice_11_0", "hi", split="test"
            )
        else:
            common_voice["train"] = load_dataset(
                "mozilla-foundation/common_voice_11_0", "hi", split="train+validation"
            ).select(range(5))
            common_voice["test"] = load_dataset(
                "mozilla-foundation/common_voice_11_0", "hi", split="test"
            ).select(range(5))

        common_voice = common_voice.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "path",
                "segment",
                "up_votes",
            ]
        )
        return common_voice