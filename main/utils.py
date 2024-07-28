import logging
import os
import warnings
import tempfile
from pathlib import Path
from typing import *

import pkg_resources  # type: ignore
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import librosa
import wandb
import plotly.graph_objs as go
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def stringify(obj: Union[Mapping, List, Tuple, Any]):
    if isinstance(obj, Mapping):
        return {k: stringify(v) for k, v in obj.items()}
    elif isinstance(obj, (List, Tuple)):
        return [stringify(v) for v in obj]
    else:
        return str(obj)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
                "datamodule",
                "model",
                "callbacks",
                "logger",
                "trainer",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger,
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    hparams["pacakges"] = get_packages_list()

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger,
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def get_packages_list() -> List[str]:
    return [f"{p.project_name}=={p.version}" for p in pkg_resources.working_set]


def retry_if_error(fn: Callable, num_attemps: int = 10):
    for attempt in range(num_attemps):
        try:
            return fn()
        except:
            print(f"Retrying, attempt {attempt + 1}")
            pass
    return fn()


class SavePytorchModelAndStopCallback(Callback):
    def __init__(self, path: str, attribute: Optional[str] = None):
        self.path = path
        self.attribute = attribute

    def on_train_start(self, trainer, pl_module):
        model, path = pl_module, self.path
        if self.attribute is not None:
            assert_message = "provided model attribute not found in pl_module"
            assert hasattr(pl_module, self.attribute), assert_message
            model = getattr(
                pl_module, self.attribute, hasattr(pl_module, self.attribute)
            )
        # Make dir if not existent
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        # Save model
        torch.save(model, path)
        log.info(f"PyTorch model saved at: {path}")
        # Stop trainer
        trainer.should_stop = True


def log_wandb_audio_batch(
        logger: WandbLogger, id: str, samples: torch.Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(num_items):
            torchaudio.save(
                tmpdir / f"{i}.wav", samples[i].detach().cpu(), sample_rate=sampling_rate
            )

        logger.log(
            {
                f"sample_{i}_{id}": wandb.Audio(
                    str(tmpdir / f"{i}.wav"),
                    caption=caption,
                    sample_rate=sampling_rate,
                )
                for idx in range(num_items)
            }
        )


def log_wandb_audio_spectrogram(
        logger: WandbLogger, id: str, samples: torch.Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )