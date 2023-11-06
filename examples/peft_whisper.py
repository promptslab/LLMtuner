from llmtuner import Tuner, Dataset, Model


# Create a dataset instance for the audio files
dataset = Dataset(dummy_data = True, type_ = "full")

#Model
model = Model("openai/whisper-large-v2", use_peft = True)

training_args_dict = {
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=500,
    num_train_epochs=1,
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=225,
    logging_steps=100,
    max_steps=4000
    }

tuner = Tuner(model, dataset, training_args_dict)
trained_model = tuner.fit()

print(tuner.wer_eval('test'))
