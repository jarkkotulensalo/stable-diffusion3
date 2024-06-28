# Stable Diffusion 3 (SD3) inference with <16GB memory

We use the 8-bit quantised version of T5-XXL encoder following the tutorial on https://huggingface.co/blog/sd3.
This allows inference in less than 16GB of memory.

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
conda env create -f env.yaml

conda activate sd3
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

| ![Example Image 1](examples/cat_wizard_gandalf_lord_of_the_rings_detailed_fant.png) | ![Example Image 2](examples/john_snow_from_game_of_thrones_is_showing_a_sign.png) |
|-----------------------------------------|-----------------------------------------|
| ![Example Image 3](examples/shrek_showing_a_sign_that_says_happy_birthday_john.png) | ![Example Image 4](examples/mickey_mouse_holding_a_sign_rahaa_on_with_a_happ.png) |
