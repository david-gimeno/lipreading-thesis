#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks import ASRTask
from src.tasks import AVSRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.model_summary import model_summary

import os
import sys
import yaml
import argparse
from tqdm import tqdm
from colorama import Fore
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import *
from src.transforms import *
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

class ModelAdaptor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, speech):
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        batch = to_device(batch, device=speech.device)

        enc, _ = self.model.encode(**batch)
        return torch.mean(torch.mean(enc, axis=-1), axis=-1)

def integrated_gradients(model, eval_sample):
    model.eval()
    model.zero_grad()

    keys_of_interest = ['speech'] # , 'speech_lengths', 'text', 'text_lengths']
    eval_sample = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in eval_sample.items()}

    inputs = tuple([
        eval_sample[key]
        for key in keys_of_interest
    ])
    baselines = tuple([
        torch.zeros_like(input_)
        for input_ in inputs
    ])

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs,
        baselines,
        n_steps=16,
        return_convergence_delta=False,
    )

    for frame_idx in range(eval_sample['speech'].shape[1]):
        attrs = attributions[0].unsqueeze(-1).detach().cpu().numpy()[0][frame_idx]
        original_image = eval_sample['speech'].unsqueeze(-1).detach().cpu().numpy()[0][frame_idx]
        plt_fig, plt_axis = viz.visualize_image_attr(
            attrs,
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed Integrated Gradients",
            use_pyplot=False,
        )
        plt_fig.savefig(os.path.join(args.output_dir, f'frame{str(frame_idx).zfill(3)}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Audio-Visual Speech Recognition System.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="", type=str, help="Path to where the dataset split is")
    parser.add_argument("--filter-spkr-ids", nargs='+', default=["all-spkrs"], type=str, help="Choose the speaker's data you want to use")

    parser.add_argument("--snr-target", default=9999, type=int, help="A specific signal-to-noise rate when adding noise to the audio waveform.")
    parser.add_argument("--noise", default="./src/noise/babble_noise.wav", type=str, help="Path to .wav file of noise")

    parser.add_argument("--config-file", required=True, type=str, help="Path to a config file that specifies the AVSR model architecture")
    parser.add_argument("--load-checkpoint", default="", type=str, help="Path to load a pretrained AVSR model")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the attribution interpretable output")

    args = parser.parse_args()

    # -- configuration architecture details
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- security checks
    security_checks(config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(config)

    # -- audio preprocessing
    eval_audio_transforms = Compose([
        AddNoise(noise_path=args.noise, sample_rate=16000, snr_target=args.snr_target),
    ])

    # -- video preprocessing
    """When using models from LRE journal paper.
    VLRF --> 50 fps, (0.392, 0.142) # and removing frame sampling from dataset
    LIP-RTVE --> 25 fps (0.491, 0.166)
    """
    fps = 25
    (mean, std) = (0.421, 0.165)

    eval_video_transforms = Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        CenterCrop((88,88)),
    ])

    # -- building AVSR end-to-end system
    task_class = AVSRTask if config.task == 'avsr' else ASRTask
    e2e = task_class.build_model(config).to(
        dtype=getattr(torch, config.dtype),
        device=config.device,
    )
    print(model_summary(e2e))

    # -- loading the AVSR end-to-end system from a checkpoint
    load_e2e(e2e, ['entire-e2e'], args.load_checkpoint, config)

    # -- wrapping model adaptor
    model = ModelAdaptor(e2e)

    # -- defining Integrated Gradients
    ig = IntegratedGradients(model)

    # -- creating dataloaders
    eval_loader = get_dataloader(config, dataset_path=args.dataset, audio_transforms=eval_audio_transforms, video_transforms=eval_video_transforms, tokenizer=tokenizer, converter=converter, filter_spkr_ids=args.filter_spkr_ids, is_training=False)
    eval_sample = next(iter(eval_loader))
    eval_sample = next(iter(eval_loader))
    eval_sample = next(iter(eval_loader))

    # -- create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    attributions = integrated_gradients(model, eval_sample)

