<div align="center">
<img width="220px" src="https://raw.githubusercontent.com/promptslab/LLMTuner/main/assets/logo.png">
<h1>LLMTuner</h1></div>
<!-- 
<h2 align="center">LLMTuner</h2> -->

<p align="center">
  <p align="center">LLMTuner: Fine-Tune Llama, Whisper, and other LLMs with best practices like LoRA, QLoRA, through a sleek, scikit-learn-inspired interface.
</p>
</p>

 <h4 align="center">
  <a href="https://github.com/promptslab/LLMTuner/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="LLMTuner is released under the Apache 2.0 license." />
  </a>
  <a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" />
  </a>
  <a href="https://discord.gg/m88xfYMbK6">
    <img src="https://img.shields.io/badge/Discord-Community-orange" alt="Community" />
  </a>
  <a href="#">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab" />
  </a>
</h4>


## Installation

### With pip

This repository is tested on Python 3.7+

You should install Promptify using Pip command

```bash
pip3 install git+https://github.com/promptslab/LLMTuner.git
```

## Quick tour

To fine a Large models we provide the `Tuner` API.

```python

from llmtuner import Tuner, Dataset, Model, Deployment

# Initialize the Whisper model with parameter-efficient fine-tuning
model = Model("openai/whisper-small", use_peft=True)

# Create a dataset instance for the audio files
dataset = Dataset('/path/to/audio_folder')

# Set up the tuner with the model and dataset for fine-tuning
tuner = Tuner(model, dataset)

# Fine-tune the model
trained_model = tuner.fit()

# Inference with Fine-tuned model
tuner.inference('sample.wav')

# Launch an interactive UI for the fine-tuned model
tuner.launch_ui('Model Demo UI')

# Set up deployment for the fine-tuned model
deploy = Deployment('aws')  # Options: 'fastapi', 'aws', 'gcp', etc.

# Launch the model deployment
deploy.launch()

```


<h2>Features 🤖 </h2>
<ul>
  <li>🏋️‍♂️ Effortless Fine-Tuning: Finetune state-of-the-art LLMs like Whisper, Llama with minimal code</li>
  <li>⚡️ Built-in utilities for techniques like LoRA and QLoRA <li>
  <li>⚡️ Interactive UI: Launch webapp demos for your finetuned models with one click</li>
  <li>🏎️ Simplified Inference: Fast inference without separate code</li>
  <li>🌐 Deployment Readiness: (Coming Soon) Deploy your models with minimal effort to aws, gcp etc, ready to share with the world.</li> 
</ul>



### Supporting wide-range of Prompt-Based NLP tasks :

| Task Name | Colab Notebook | Status |
|-------------|-------|-------|
| Fine-Tune Whisper | [Fine-Tune Whisper](https://colab.research.google.com/drive/1j_1AcPRk4s1uivVRSwrsOfodjPv55Jpc?usp=sharing) | ✅  |
| Fine-Tune Whisper Quantized  | [LoRA](https://colab.research.google.com/drive/1j_1AcPRk4s1uivVRSwrsOfodjPv55Jpc?usp=sharing) | ✅    |
| Fine-Tune Llama | [Coming soon..](#) | ✅    |


## Community 
<div align="center">
If you are interested in Fine-tuning Open source LLMs, Building scalable Large models, Prompt-Engineering, and other latest research discussions, please consider joining <a href="https://discord.gg/m88xfYMbK6">PromptsLab</a></div>
<div align="center">
<img alt="Join us on Discord" src="https://img.shields.io/discord/1069129502472556587?color=5865F2&logo=discord&logoColor=white">
</div>

```

@misc{LLMtuner2023,
  title = {LLMTuner: Fine-Tune Llama, Whisper, and other Large Models with best practices like LoRA, QLoRA, through a sleek, scikit-learn-inspired interface.},
  author = {Pal, Ankit},
  year = {2023},
  howpublished = {\url{https://github.com/promptslab/LLMtuner}}
}

```

## 💁 Contributing

We welcome any contributions to our open source project, including new features, improvements to infrastructure, and more comprehensive documentation. 
Please see the [contributing guidelines](#)
