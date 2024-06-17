# Stable Diffusion 3 (SD3) inference example

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
conda env create -f env.yaml

conda activate stable-diffusion3
```

Then run
```bash
python src/generate_sd3.py
```

### Prompts
You can run multiple prompts by specifying them on configs/prompts.txt

```
cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k
Shrek showing a sign that says 'Happy birthday John' to Donkey.
```

### config
Other configs are specified in configs/sd3.yaml

### output
You can find the generated images at output folder
