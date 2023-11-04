import uuid
class BaseTrainer:
    def __init__(self, output_dir, task):

        if output_dir:
            self.output_dir = output_dir
        else:
            self.uuid_d = str(uuid.uuid4())
            self.output_dir = f"model_{self.uuid_d}"
        
        self.task = task

    def setup_trainer(self):
        # Implementation for basic setup
        pass

    def start_training(self):
        # Basic training loop implementation
        pass

    def push_to_hub(self):
        # Basic implementation for pushing to the hub
        pass


