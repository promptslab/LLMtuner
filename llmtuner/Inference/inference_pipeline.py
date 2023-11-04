import torch
from transformers import (
    WhisperForConditionalGeneration,
    AutomaticSpeechRecognitionPipeline,
    WhisperTokenizer,
    WhisperProcessor,
    pipeline
)
from peft import PeftModel, PeftConfig
import evaluate
WER_score = evaluate.load("wer")

class BaseTranscriptionPipeline:
    def transcribe(self, audio):
        raise NotImplementedError

    def transcribe_bulk(self, audio_files):
        return [self.transcribe(audio) for audio in audio_files]

    def wer_score(self, audio_path, ground_truth):
        result = self.transcribe(audio_path)
        wer = 100 * WER_score.compute(predictions=[result], references=[ground_truth])
        return {"wer": wer}

    def wer_score_bulk(self, audio_paths, ground_truths):
        results = self.transcribe_bulk(audio_paths)
        wer = 100 * WER_score.compute(predictions=results, references=ground_truths)
        return {"wer": wer}

    def evaluate_dataset(self, dataset):
        audio_paths = [item['audio']['path'] for item in dataset]
        sentences = [item['sentence'] for item in dataset]
        return self.wer_score_bulk(audio_paths, sentences)

class ASRTranscriptionPipeline(BaseTranscriptionPipeline):
    def __init__(self, model_id, language, task):
        peft_config = PeftConfig.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_id)
        tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        self.feature_extractor = processor.feature_extractor
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        self.pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=self.feature_extractor)

    def transcribe(self, audio):
        with torch.cuda.amp.autocast():
            text = self.pipe(audio, generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids}, max_new_tokens=255)["text"]
        return text



class GenericTranscriptionPipeline(BaseTranscriptionPipeline):
    def __init__(self, model_name):
        self.pipe = pipeline(model=model_name)

    def transcribe(self, audio_path):
        result = self.pipe(audio_path)
        return result.get('text', '')

class WhisperEval:
    def __init__(self, model_id_or_name, language, task, use_peft=False):
        self.pipeline = ASRTranscriptionPipeline(model_id_or_name, language, task) if use_peft else GenericTranscriptionPipeline(model_id_or_name)

    def transcribe(self, audio):
        return self.pipeline.transcribe(audio)

    def transcribe_bulk(self, audio_files):
        return self.pipeline.transcribe_bulk(audio_files)

    def wer_score(self, audio_path, ground_truth):
        return self.pipeline.wer_score(audio_path, ground_truth)

    def wer_score_bulk(self, audio_paths, ground_truths):
        return self.pipeline.wer_score_bulk(audio_paths, ground_truths)

    def evaluate_dataset(self, dataset):
        return self.pipeline.evaluate_dataset(dataset)