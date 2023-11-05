from datasets import load_dataset, DatasetDict

class Dataset:
    def __init__(self, train_dir=None, test_dir=None, dummy_data=False, type_='full', n_samples = 5):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.dummy_data = dummy_data
        self.type_ = type_
        self.n_samples = n_samples
        self.dataset = None

        if self.dummy_data:
            self.dataset = self.load_dummy_data(self.type_, self.n_samples)
        else:
            self.dataset = self.load_local_datasets()

    def load_local_datasets(self):
        """Load datasets from local directories."""
        hf_dataset = DatasetDict()
        hf_dataset["train"] = load_dataset("audiofolder", data_dir=self.train_dir, split="train")
        hf_dataset["test"] = load_dataset("audiofolder", data_dir=self.test_dir, split="train")
        return hf_dataset
    
    def load_dummy_data(self, type_="full", n_samples = 5):
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
            ).select(range(n_samples))
            common_voice["test"] = load_dataset(
                "mozilla-foundation/common_voice_11_0", "hi", split="test"
            ).select(range(n_samples))

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