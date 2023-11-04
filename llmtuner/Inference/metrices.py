class WERMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = self._load_metric()

    def _load_metric(self):
        try:
            import evaluate
            return evaluate.load("wer")
        except ImportError:
            raise ImportError("Ensure you have the 'evaluate' library installed.")

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # decode predictions and labels

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # compute WER
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}