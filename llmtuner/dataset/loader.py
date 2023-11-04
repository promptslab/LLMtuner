from datasets import load_dataset, DatasetDict

class Dataset:
    def __init__(self, train_dir=None, test_dir=None, dummy_data=False, type_='full'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.dummy_data = dummy_data
        self.type_ = type_
        self.dataset = None

        if self.dummy_data:
            self.dataset = self.load_dummy_data()
        else:
            self.dataset = self.load_local_datasets()

    def load_local_datasets(self):
        """Load datasets from local directories."""
        hf_dataset = DatasetDict()
        hf_dataset["train"] = load_dataset("audiofolder", data_dir=self.train_dir, split="train")
        hf_dataset["test"] = load_dataset("audiofolder", data_dir=self.test_dir, split="train")
        return hf_dataset
    
    def load_dummy_data(self):
        """Load dummy data for testing purposes."""
        common_voice = DatasetDict()
        common_voice_split = "train+validation" if self.type_ == 'full' else "train[:5]+validation[:5]"
        common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split=common_voice_split)
        common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test[:5]")

        columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
        common_voice["train"] = common_voice["train"].remove_columns(columns_to_remove)
        common_voice["test"] = common_voice["test"].remove_columns(columns_to_remove)
        return common_voice