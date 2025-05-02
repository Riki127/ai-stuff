# How to run

## Virtual enviroment for python
Follow the steps in this link https://huggingface.co/docs/datasets/en/installation
For windows when you want to activate run the PowerShell script generated
```
.env\Scripts\Activate.ps1
```


For VS code 
Press Ctrl+Shift+P (or Cmd+Shift+P on macOS)

Type: Python: Select Interpreter

And choose the one from .env
## Download microsoft c++ build tools (Windows only)
Download the c++ build tools https://visualstudio.microsoft.com/visual-cpp-build-tools/

This is needed for the llama-cpp package

## Install dependencies with pip
```
pip install datasets
pip install sentence-transformers
pip install chromadb
pip install llama-cpp-python
```
## Download the LLM
Download this specific LLM https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

And add it to /models folder, or any folder that would suit best just change the model_path in the code
