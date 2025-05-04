## Preparation
python -m pip install --upgrade setuptools wheel twine check-wheel-contents #TPU v5e/v5p/v6e only \
bash setup.sh \
python -m pip install --upgrade gradio 
## Step 1
### Edit f5.yml
pretrained_model_name_or_path: '/xxx/model_xxx.pt' #place F5 model here \
vocab_name_or_path: '/xxx/vocab.txt' #place vocab here \
compiled_path: '/xxx/jax-F5-TTS/' #AOT compiled file path
## Step 2
### Do AOT Compiling
python -m src.maxdiffusion.generate_f5_aot src/maxdiffusion/configs/f5.yml
## Step 3
### Load AOT File And Open Gradio
python -m src.maxdiffusion.f5_gradio_ui_load_aot src/maxdiffusion/configs/f5.yml