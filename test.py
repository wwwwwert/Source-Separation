import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import hw_ss.model as module_model
from hw_ss.metric import pesq_metric, si_sdr_metric
from hw_ss.trainer import Trainer
from hw_ss.utils import ROOT_PATH
from hw_ss.utils.object_loading import get_dataloaders
from hw_ss.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # setup data_loader instances
    dataloaders, _ = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=251)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []

    pesq_loss = pesq_metric.PESQMetric()
    si_sdr_loss = si_sdr_metric.SI_SDR()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model trainable parameters:', pytorch_total_params)

    pesqs = []
    si_sdrs = []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            batch.update(output)
            for i in range(len(batch["preds"])):
                pesq = pesq_loss(batch["preds"].cpu(), batch["target_audio"].cpu())
                si_sdr = si_sdr_loss(batch["preds"].cpu(), batch["target_audio"].cpu())
                results.append(
                    {
                        "mix_length": batch["audio_length"][i].item(),
                        "ref_length": batch["ref_length"][i].item() / 16000,
                        "pesq": pesq.item(),
                        "si_sdr": si_sdr.item()
                    }
                )
                pesqs.append(pesq)
                si_sdrs.append(si_sdr)
    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)

    print('Mean PESQ:', np.mean(pesqs))
    print('Mean SI-SDR:', np.mean(si_sdrs))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
