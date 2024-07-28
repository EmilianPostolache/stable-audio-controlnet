# Stable Audio ControlNet

Fine-tune Stable Audio Open with DiT ControlNet. On 16GB VRAM GPU you can use adapter of 20% the size of the full DiT with bs=1
and mixed fp16. Inference code coming soon. **Very experimental at the moment.**

**Work in progress!**

# TODO
- [x] Add training code.
- [x] Add generation code.
- [ ] Improve inference demo.

#  Demo 
In the following we detail training a model for compositional music generation (edit, accompaniments, separation),
on MusDB.

## Setup
After installing the requirements copy `.env.tmp` as `.env` and replace with your own variables (example values are random):

```
DIR_LOGS=/logs
DIR_DATA=/data

# Required if using wandb logger
WANDB_PROJECT=audioproject
WANDB_ENTITY=johndoe
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx
```

## Dataset (MusDB18HQ)

First download the sharded version of [MusDB18HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) from https://drive.google.com/drive/folders/1bwiJbRH_0BsxGFkH0No-Rg_RHkVR2gc7?usp=sharing
and put the files `test.tar` and `train.tar` inside `data/musdb18hq/`.

## Train

For training run
```
PYTHONUNBUFFERED=1 TAG=musdb-controlnet python train.py exp=train_musdb_controlnet \
datamodule.train_dataset.path=data/musdb18hq/train.tar \ 
datamodule.val_dataset.path=data/musdb18hq/test.tar
```

# Credits

- Evans, Z., Parker, J. D., Carr, C. J., Zukowski, Z., Taylor, J., & Pons, J. (2024). Stable Audio Open. arXiv preprint arXiv:2407.14358.
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3836-3847).
- Chen, J., Wu, Y., Luo, S., Xie, E., Paul, S., Luo, P., ... & Li, Z. (2024). Pixart-{\delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252.
