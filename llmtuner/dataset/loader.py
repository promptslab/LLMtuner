from datasets import load_dataset, concatenate_datasets, DatasetDict

class Dataset:
    def __init__(self, train_dir = None, test_dir = None, dummy_data = False, type_ = 'full'):

        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.dummy_data = dummy_data
        self.type_ = 'full'

    def load_local_datasets(self, train_dir="train_data/", test_dir="test_data/"):
        """
        Load datasets from local directories, combine them, and split based on the provided ratio.
        
        Parameters:
        - train_dir (str): Path to the training data directory. Default is "train_data/".
        - test_dir (str): Path to the testing data directory. Default is "test_data/".
        - split_ratio (float): Ratio to split the combined dataset into training and testing. Default is 0.3.
        
        Returns:
        DatasetDict: A dictionary containing the train and test datasets.
        """
        hf_dataset = DatasetDict()
        hf_dataset["train"] = load_dataset("audiofolder", data_dir=train_dir, split="train")
        hf_dataset["test"]  = load_dataset("audiofolder", data_dir=test_dir, split="train")
        return hf_dataset
    
    def load_dummy_data(self, type_ = 'full'):
        """
        Load dummy data from the `mozilla-foundation/common_voice_11_0` dataset.
        
        Returns:
        DatasetDict: A dictionary containing the train and test datasets with selected columns removed.
        """
        common_voice = DatasetDict()

        if type_ == 'full':
            common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", 
                                                 split="train+validation")
            common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", 
                                                split="test")
        else:
            common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", 
                                                 split="train+validation").select(range(5))
            common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", 
                                                split="test").select(range(5))
            
        common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", 
                                                    "gender", "locale", "path", "segment", "up_votes"])
        return common_voice