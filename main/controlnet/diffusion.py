import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.diffusion import UNetCFG1DWrapper, UNet1DCondWrapper, \
    ConditionedDiffusionModelWrapper, DiTWrapper
from torch import nn
import numpy as np
import typing as tp

from stable_audio_tools.models.conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from main.controlnet.dit import DiffusionTransformer
from stable_audio_tools.models.factory import create_pretransform_from_config
from stable_audio_tools.models.pretransforms import Pretransform


from main.controlnet.controlnet import ControlNetDiffusionTransformer

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                supports_controlnet_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond
        self.supports_controlnet_cond = supports_controlnet_cond


    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                controlnet_cond: torch.Tensor = None,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

class DiTControlNetWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        controlnet_depth_factor,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = DiffusionTransformer(*args, **kwargs)
        kwargs["depth"] = int(controlnet_depth_factor * kwargs["depth"])
        self.controlnet = ControlNetDiffusionTransformer(*args, **kwargs)
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #        param *= 0.5

    def forward(self,
                x,
                t,
                controlnet_cond=None,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                scale_controlnet_embeds: float = 1.0,
                scale_controlnet_cond: float = 1.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        controlnet_embeds, cfg_cross_attn_dropout_mask, cfg_prepend_dropout_mask = self.controlnet(x,
                                            t,
                                            controlnet_cond=scale_controlnet_cond * controlnet_cond,
                                            cross_attn_cond=cross_attn_cond,
                                            cross_attn_cond_mask=cross_attn_mask,
                                            negative_cross_attn_cond=negative_cross_attn_cond,
                                            negative_cross_attn_mask=negative_cross_attn_mask,
                                            input_concat_cond=input_concat_cond,
                                            prepend_cond=prepend_cond,
                                            prepend_cond_mask=prepend_cond_mask,
                                            cfg_scale=cfg_scale,
                                            cfg_dropout_prob=cfg_dropout_prob,
                                            global_embed=global_cond,
                                            **kwargs)
        controlnet_embeds = [controlnet_embed * scale_controlnet_embeds for controlnet_embed in controlnet_embeds]

        return self.model(
            x,
            t,
            controlnet_embeds=controlnet_embeds,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_cross_attn_dropout_mask=cfg_cross_attn_dropout_mask,
            cfg_prepend_dropout_mask=cfg_prepend_dropout_mask,
            scale_phi=scale_phi,
            global_embed=global_cond,
            **kwargs)


class ConditionedControlNetDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            controlnet_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.controlnet_cond_ids = controlnet_cond_ids
        self.min_input_length = min_input_length

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        controlnet_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if len(self.controlnet_cond_ids) > 0:
            controlnet_conds = []
            for key in self.controlnet_cond_ids:
                controlnet_conds.append(conditioning_tensors[key][0].unsqueeze(1))
            controlnet_cond = torch.cat(controlnet_conds, dim=1).sum(dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask,
                "controlnet_cond": controlnet_cond
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        conditioning_inputs = self.get_conditioning_inputs(cond)
        return self.model(x, t, **conditioning_inputs, **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)



def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):
    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'adp_cfg_1d':
        diffusion_model = UNetCFG1DWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'adp_1d':
        diffusion_model = UNet1DCondWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit_controlnet':
        diffusion_model = DiTControlNetWrapper(**diffusion_model_config)

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])
    controlnet_cond_ids = diffusion_config.get('controlnet_cond_ids', [])

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size
    elif diffusion_model_type == "dit_controlnet":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        wrapper_fn = ConditionedDiffusionModelWrapper

        extra_kwargs["diffusion_objective"] = diffusion_objective

    elif model_type == "diffusion_prior":
        prior_type = model_config.get("prior_type", None)
        assert prior_type is not None, "Must specify prior_type in diffusion prior model config"

        if prior_type == "mono_stereo":
            from stable_audio_tools.models.diffusion_prior import MonoToStereoDiffusionPrior
            wrapper_fn = MonoToStereoDiffusionPrior
    elif model_type == "diffusion_cond_controlnet":
        wrapper_fn = ConditionedControlNetDiffusionModelWrapper
        assert diffusion_model_type == 'dit_controlnet'

    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        controlnet_cond_ids=controlnet_cond_ids,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        **extra_kwargs
    )