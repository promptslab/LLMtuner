from llmtuner import InstructionDataset, Model, Tuner, Estimater, UI

# Load the dataset
instruction_dataset = InstructionDataset("./alpaca_data")

# Initialize the model

model = Model.load("llama_lora")
tuner = Tuner(parameters)

# Finetune the model
tuner = tuner.fit(model = "", dataset = "")

# Perform inference
output = model.inference(texts=["Why LLM models are becoming so important?"])

# Save the model
model.save("llama_lora")

#calcuate estimate cost to deply the model
Estimater(model)

#play with model on UI
#UI(model)

# deploy the model and return the endpoint
endpoint = model.deploy("llama_lora", "s3://llama_lora")

# example of endpoint 
# endpoint = "https://llama-lora-1234.execute-api.us-east-1.amazonaws.com/llama-lora"


# use endpoint to perform inference using request module
# example of request
# data = {"texts": ["Why LLM models are becoming so important?"]}
# response = requests.post(endpoint, json=data)
# print(response.json())

# use endpoint to perform inference using curl
# example of curl
# curl -X POST -H "Content-Type: application/json" -d '{"texts": ["Why LLM models are becoming so important?"]}' https://llama-lora-1234.execute-api.us-east-1.amazonaws.com/llama-lora

# delete the endpoint
# model.delete("llama_lora")

# list the items from model building to deployment and more
# base model loading
# tuner loading
# finetuning
# inference
# model saving
# model deployment
# model inference

