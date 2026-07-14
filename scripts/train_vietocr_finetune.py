from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.services.metrics import cer, wer


def patch_numpy_for_imgaug() -> None:
    import numpy as np

    if hasattr(np, "sctypes"):
        pass
    else:
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_],
        }

    original_fromstring = np.fromstring

    def fromstring_compat(string, dtype=float, count=-1, *, sep="", like=None):  # type: ignore[no-untyped-def]
        if isinstance(string, (bytes, bytearray, memoryview)) and sep == "":
            return np.frombuffer(string, dtype=dtype, count=count)
        kwargs = {"dtype": dtype, "count": count, "sep": sep}
        if like is not None:
            kwargs["like"] = like
        return original_fromstring(string, **kwargs)

    np.fromstring = fromstring_compat  # type: ignore[assignment]


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_vietocr_config(config_name: str, config_file: Path | None = None):
    from vietocr.tool.config import Cfg

    if config_file and config_file.exists():
        return Cfg.load_config_from_file(str(config_file))
    return Cfg.load_config_from_name(config_name)


def read_annotation(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            image, label = line.split("\t", 1)
            rows.append((image, label))
            if limit and len(rows) >= limit:
                break
    return rows


def evaluate_model(config: dict[str, Any], dataset_dir: Path, val_annotation: Path, sample_limit: int) -> dict[str, Any]:
    from vietocr.tool.predictor import Predictor

    rows = read_annotation(val_annotation, sample_limit)
    if not rows:
        return {"sample_count": 0, "cer": None, "wer": None, "examples": []}

    predictor = Predictor(config)
    truth_texts: list[str] = []
    pred_texts: list[str] = []
    examples: list[dict[str, str]] = []
    for image_rel, truth in rows:
        image_path = dataset_dir / image_rel
        with Image.open(image_path) as image:
            pred = str(predictor.predict(image.convert("RGB"))).strip()
        truth_texts.append(truth)
        pred_texts.append(pred)
        if len(examples) < 8:
            examples.append({"image": image_rel, "truth": truth, "pred": pred})

    pred_joined = "\n".join(pred_texts)
    truth_joined = "\n".join(truth_texts)
    return {
        "sample_count": len(rows),
        "cer": cer(pred_joined, truth_joined),
        "wer": wer(pred_joined, truth_joined),
        "examples": examples,
    }


def configure_training(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_annotation = dataset_dir / args.train_annotation
    val_annotation = dataset_dir / args.val_annotation
    if not train_annotation.exists():
        raise RuntimeError(f"Missing train annotation: {train_annotation}")

    config = load_vietocr_config(args.config_name, args.config_file)
    device = resolve_device(args.device)
    pin_memory = bool(device.startswith("cuda"))

    config["device"] = device
    config["dataset"]["name"] = "congvan"
    config["dataset"]["data_root"] = str(dataset_dir)
    config["dataset"]["train_annotation"] = args.train_annotation
    config["dataset"]["valid_annotation"] = args.val_annotation if val_annotation.exists() else None
    config["dataset"]["image_height"] = args.image_height
    config["dataset"]["image_min_width"] = args.image_min_width
    config["dataset"]["image_max_width"] = args.image_max_width

    config["trainer"]["batch_size"] = args.batch_size
    config["trainer"]["iters"] = args.iters
    config["trainer"]["print_every"] = args.print_every
    config["trainer"]["valid_every"] = args.valid_every
    config["trainer"]["metrics"] = args.eval_samples
    config["trainer"]["export"] = str(output_dir / "transformerocr.pth")
    config["trainer"]["checkpoint"] = str(output_dir / "checkpoint.pth")
    config["trainer"]["log"] = str(output_dir / "train.log")

    config["optimizer"]["max_lr"] = args.max_lr
    config["optimizer"]["pct_start"] = args.pct_start
    config["dataloader"]["num_workers"] = args.num_workers
    config["dataloader"]["pin_memory"] = pin_memory
    config["aug"]["image_aug"] = args.image_aug
    config["aug"]["masked_language_model"] = args.masked_language_model
    config["predictor"]["beamsearch"] = False
    config["quiet"] = False

    init_weights = args.init_weights
    if init_weights and init_weights.exists():
        config["pretrain"] = str(init_weights.resolve())
    elif args.resume_existing and (output_dir / "transformerocr.pth").exists():
        config["pretrain"] = str((output_dir / "transformerocr.pth").resolve())

    return config, dataset_dir, output_dir


def train(args: argparse.Namespace) -> dict[str, Any]:
    patch_numpy_for_imgaug()
    from vietocr.model.trainer import Trainer

    config, dataset_dir, output_dir = configure_training(args)
    cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        if args.rebuild_lmdb:
            for name in ("train_congvan", "valid_congvan"):
                path = output_dir / name
                if path.exists():
                    shutil.rmtree(path)
        trainer = Trainer(config, pretrained=True)
        trainer.train()
        # VietOCR's Trainer writes the best validation checkpoint to
        # ``trainer.export``. Preserve it instead of overwriting it with the
        # final iteration, which may already have regressed/overfit.
        best_weights = Path(config["trainer"]["export"])
        last_weights = output_dir / "transformerocr_last.pth"
        trainer.save_weights(str(last_weights))
        if not best_weights.exists():
            shutil.copy2(last_weights, best_weights)
    finally:
        os.chdir(cwd)

    inference_config = dict(config)
    inference_config["weights"] = str((output_dir / "transformerocr.pth").resolve())
    inference_config_path = output_dir / "config.yml"
    from vietocr.tool.config import Cfg

    Cfg(inference_config).save(str(inference_config_path))

    eval_summary = evaluate_model(
        inference_config,
        dataset_dir,
        dataset_dir / args.val_annotation,
        args.eval_samples,
    )
    summary = {
        "model_dir": str(output_dir),
        "weights": inference_config["weights"],
        "last_weights": str((output_dir / "transformerocr_last.pth").resolve()),
        "config": str(inference_config_path),
        "dataset_dir": str(dataset_dir),
        "device": inference_config["device"],
        "iters": args.iters,
        "batch_size": args.batch_size,
        "eval": eval_summary,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VietOCR on Vietnamese administrative document line crops.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_template" / "vietocr_finetune")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "vietocr-congvan")
    parser.add_argument("--config-name", default="vgg_transformer")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--train-annotation", default="train.txt")
    parser.add_argument("--val-annotation", default="val.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--valid-every", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--image-min-width", type=int, default=32)
    parser.add_argument("--image-max-width", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--rebuild-lmdb", action="store_true")
    parser.add_argument("--image-aug", action="store_true")
    parser.add_argument(
        "--masked-language-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable masked-language-model augmentation (use --no-masked-language-model to disable).",
    )
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    printable = {
        "model_dir": summary["model_dir"],
        "weights": summary["weights"],
        "config": summary["config"],
        "device": summary["device"],
        "eval": summary["eval"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
