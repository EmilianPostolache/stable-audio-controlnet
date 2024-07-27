import json

import torch
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.models.utils import load_ckpt_state_dict

from main.controlnet.factory import create_model_from_config

from huggingface_hub import hf_hub_download



def get_pretrained_controlnet_model(name: str, depth_factor=0.5):
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')

    with open(model_config_path) as f:
        model_config = json.load(f)
    model_config["model_type"] = "diffusion_cond_controlnet"
    model_config["controlnet_depth_factor"] = depth_factor

    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    state_dict = load_ckpt_state_dict(model_ckpt_path)

    model.load_state_dict(state_dict, strict=False)
    state_dict_controlnet = {k.split('model.model.')[-1]: v for k, v in state_dict.items() if k.startswith('model.model')}
    model.controlnet.model.load_state_dict(state_dict_controlnet, strict=False)

    return model, model_config


if __name__ == '__main__':
    from main.data.dataset_moisesdb import create_moisesdb_dataset, collate_fn_conditional
    from torch.utils.data import DataLoader

    model, model_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0")
    model = model.cuda()

