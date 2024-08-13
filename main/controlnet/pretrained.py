import json
from typing import List

import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.models.utils import load_ckpt_state_dict

from main.controlnet.factory import create_model_from_config

from huggingface_hub import hf_hub_download



def get_pretrained_controlnet_model(name: str,
                                    controlnet_types : List[str],
                                    depth_factor=0.5):
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')

    with open(model_config_path) as f:
        model_config = json.load(f)
    model_config["model_type"] = "diffusion_cond_controlnet"
    model_config["model"]["diffusion"]['config']["controlnet_depth_factor"] = depth_factor
    model_config["model"]["diffusion"]["type"] = "dit_controlnet"

    model_config["model"]["diffusion"]['controlnet_cond_ids'] = []
    for controlnet_type in controlnet_types:
        if controlnet_type in ["audio", "envelope", "chroma"]:
            controlnet_conditioner_config = {"id": controlnet_type,
                                             "type": "pretransform",
                                             "config": {"sample_rate": model_config["sample_rate"],
                                                        "output_dim": model_config["model"]["pretransform"]["config"]["latent_dim"],
                                                       "pretransform_config": model_config["model"]["pretransform"]}}
            model_config["model"]['conditioning']['configs'].append(controlnet_conditioner_config)
            model_config["model"]["diffusion"]['controlnet_cond_ids'].append(controlnet_type)

    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    state_dict = load_ckpt_state_dict(model_ckpt_path)

    model.load_state_dict(state_dict, strict=False)
    state_dict_controlnet = {k.split('model.model.')[-1]: v for k, v in state_dict.items() if k.startswith('model.model')}
    model.model.controlnet.load_state_dict(state_dict_controlnet, strict=False)

    for controlnet_type in controlnet_types:
        if controlnet_type in ["audio", "envelope", "chroma"]:
            state_dict_pretransform = {k: v for k, v in state_dict.items() if k.startswith('pretransform.')}
            model.conditioner.conditioners[controlnet_type].load_state_dict(state_dict_pretransform)

    return model, model_config


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from main.data.dataset_musdb import create_musdb_dataset, collate_fn_conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0")
    model = model.cuda()

    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]

    dataset = create_musdb_dataset("../../data/musdb18hq/train.tar",
                                      sample_rate=44100,
                                      chunk_dur=47.57)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            pin_memory=False,
                            drop_last=True,
                            collate_fn=collate_fn_conditional,
                            num_workers=0)
    data_x, data_y, data_z = next(iter(dataloader))

    conditioning = [{
        "audio": data_y.unsqueeze(1).repeat_interleave(2, dim=1).cuda(),
        "prompt": data_z[0],
        "seconds_start": 0,
        "seconds_total": 40
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
         model,
         steps=100,
         cfg_scale=7,
         conditioning=conditioning,
         sample_size=sample_size,
         sigma_min=0.3,
         sigma_max=500,
         sampler_type="dpmpp-3m-sde",
         device=device
    )

