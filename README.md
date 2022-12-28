<h1 align="center">Nudifier</h1>
<p align="center">
  Available for <strong>Windows, Linux and Mac</strong>.
</p>
<p align="center">
  <img src="https://badgen.net/github/license/giladleef/Nudifier" />
  </a>
</p>

## About

Nudifier is a fork of the [Dreampower algorithm](https://github.com/opendreamnet/dreampower) which is itself a fork of the [DeepNude algorithm](https://github.com/stacklikemind/deepnude_official) that allows to generates fake nudes images from user-given dressed images.

It consists of several algorithms that together create a fake nude from a photo.

## Features

|                        | Nudifier   | DeepNude |
| ---------------------- | ---------- | -------- |
| Multiplatform          | ✔️          | ❌        |
| NVIDIA GPU Support     | ✔️          | ❌        |
| Multithreading         | ✔️          | ❌        |
| Automatic Scale        | ✔️          | ❌        |
| GIF Support            | ✔️          | ❌        |
| Video Support          | ✔️          | ❌        |
| Body Customization     | ✔️          | ❌        |
| Custom Masks           | ✔️          | ❌        |

## Requirements

- 64 bits CPU & operating system
- **12 GB** of RAM.

### GPU (Optional)

- **NVIDIA GPU** (AMD GPU's are not supported)
- Minimum [3.5 CUDA compute capability](https://developer.nvidia.com/cuda-gpus). (GeForce GTX 780+)
- [Latest NVIDIA drivers](https://www.nvidia.com/Download/index.aspx).
- **6 GB** of GPU VRAM.
- **8 GB** of RAM.

## Usage

To get more information about the usage, use the follwoing command:
```
python main.py run --help
```

This will print out help on the parameters the algorithm accepts.

> **The input image should be 512px * 512px in size** (parameters are provided to auto resize/scale your input).

---

# How does Nudifier work?

The algorithm uses a slightly modified version of the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) GAN architecture.

A GAN network can be trained using both **paired** and **unpaired** dataset. Paired datasets get better results and are the only choice if you want to get photorealistic results, but there are cases in which these datasets do not exist and they are impossible to create. A database in which a person appears both naked and dressed, in the same position, is extremely difficult to achieve, if not impossible.

We overcome the problem using a different approach. Instead of trying to fix the big problem, we divided the problem into 3 simpler sub-problems:

- 1. Generation of a mask that selects clothes, so we can process these pixels from the image.
- 2. Generation of a abstract representation of anatomical attributes, based on the given image position.
- 3. Generation of the fake nude image/video, by replacing the mask with the generated representation.

Using this approach, we can simply create a dataset of many naked and dressed images, and train the models on them, without needing to have the save position in both dressed and naked, but using similer positions from our dataset.
