from setuptools import setup, find_packages

setup(
    name='LLMtuner',
    version='0.1.0',
    author='Aaditya Ura (Ankit Pal)',
    author_email='aadityaura@gmail.com',
    description='A tool for tuning Large Models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/promptslab/LLMTuner',
    packages=find_packages(),
    install_requires=['datasets', 'transformers', 'librosa', 'evaluate', 'jiwer', 'gradio==3.37', 'accelerate', 'wandb', 'peft', 'bitsandbytes'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='language-model tuning hyperparameters llama whisper',
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'llmtuner=llmtuner:main',
        ],
    },
)
