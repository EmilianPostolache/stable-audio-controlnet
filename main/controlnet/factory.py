

def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'diffusion_cond_controlnet':
        from main.controlnet.diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
