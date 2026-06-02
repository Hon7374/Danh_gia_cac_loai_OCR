from __future__ import annotations

import argparse
import inspect
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image


FIELD_ALIASES = {
    "so_ky_hieu": "so_ky_hieu",
    "so": "so_ky_hieu",
    "document_number": "so_ky_hieu",
    "doc_number_symbol": "so_ky_hieu",
    "ngay_ban_hanh": "ngay_ban_hanh",
    "ngay": "ngay_ban_hanh",
    "issued_date": "ngay_ban_hanh",
    "place_date": "ngay_ban_hanh",
    "trich_yeu": "trich_yeu",
    "subject": "trich_yeu",
    "title": "trich_yeu",
    "doc_subject": "trich_yeu",
    "co_quan_ban_hanh": "co_quan_ban_hanh",
    "issuing_agency": "co_quan_ban_hanh",
    "issuer": "co_quan_ban_hanh",
    "issue_org_name": "co_quan_ban_hanh",
    "issue_org_superior": "co_quan_ban_hanh",
    "noi_gui": "noi_gui",
    "sender": "noi_gui",
    "noi_nhan": "noi_nhan",
    "receiver": "noi_nhan",
    "recipient": "noi_nhan",
    "addressee": "noi_nhan",
    "recipients": "noi_nhan",
    "loai_van_ban": "loai_van_ban",
    "document_type": "loai_van_ban",
}
FIELDS = [
    "so_ky_hieu",
    "ngay_ban_hanh",
    "trich_yeu",
    "co_quan_ban_hanh",
    "noi_gui",
    "noi_nhan",
    "loai_van_ban",
]
LABELS = ["O"] + [f"{prefix}-{field}" for field in FIELDS for prefix in ("B", "I")]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_label(label: str) -> str:
    raw = str(label or "").strip()
    if not raw or raw.upper() == "O":
        return "O"
    prefix, sep, name = raw.partition("-")
    prefix = prefix.upper()
    if sep and prefix in {"B", "I"}:
        field_name = name
    else:
        prefix = "B"
        field_name = raw
    field = FIELD_ALIASES.get(_normalize_name(field_name))
    if not field:
        raise ValueError(f"Unsupported label: {label!r}")
    return f"{prefix}-{field}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("image", "words", "boxes", "labels"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing {key!r}")
            if not (len(record["words"]) == len(record["boxes"]) == len(record["labels"])):
                raise ValueError(f"{path}:{line_no} words/boxes/labels length mismatch")
            record["_base_dir"] = str(path.parent)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def normalize_boxes(boxes: list[list[int]], width: int, height: int, already_normalized: bool) -> list[list[int]]:
    if already_normalized:
        return [[max(0, min(1000, int(coord))) for coord in box] for box in boxes]
    normalized = []
    for x0, y0, x1, y1 in boxes:
        normalized.append(
            [
                max(0, min(1000, int(1000 * x0 / max(1, width)))),
                max(0, min(1000, int(1000 * y0 / max(1, height)))),
                max(0, min(1000, int(1000 * x1 / max(1, width)))),
                max(0, min(1000, int(1000 * y1 / max(1, height)))),
            ]
        )
    return normalized


def make_training_args(args: argparse.Namespace, has_eval: bool):
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": [],
        "remove_unused_columns": False,
    }
    if "no_cuda" in params:
        kwargs["no_cuda"] = args.cpu
    elif "use_cpu" in params:
        kwargs["use_cpu"] = args.cpu
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[strategy_key] = "epoch" if has_eval else "no"
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for Vietnamese official-document fields.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/layoutlmv3-congvan-token-classification"))
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boxes-normalized", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForTokenClassification, AutoProcessor, Trainer, default_data_collator, set_seed

    class WeightedTokenTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch

            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device)
            weights[0] = 0.05
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    set_seed(args.seed)
    train_records = load_jsonl(args.train_jsonl)
    eval_records = load_jsonl(args.eval_jsonl) if args.eval_jsonl else []
    processor = AutoProcessor.from_pretrained(args.base_model, apply_ocr=False)

    def encode(example: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(example["image"])
        if not image_path.is_absolute():
            image_path = Path(example["_base_dir"]) / image_path
        image = Image.open(image_path).convert("RGB")
        words = [str(word) for word in example["words"]]
        boxes = normalize_boxes(example["boxes"], image.width, image.height, args.boxes_normalized)
        word_labels = [LABEL2ID[normalize_label(label)] for label in example["labels"]]
        encoded = processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        return {
            key: value[0] if isinstance(value, list) and len(value) == 1 else value
            for key, value in encoded.items()
        }

    train_dataset = Dataset.from_list(train_records)
    if eval_records:
        eval_dataset = Dataset.from_list(eval_records)
    elif len(train_dataset) > 1 and args.eval_ratio > 0:
        split = train_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        eval_dataset = None

    train_dataset = train_dataset.map(encode, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(encode, remove_columns=eval_dataset.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    training_args = make_training_args(args, eval_dataset is not None)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": default_data_collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor
    trainer = WeightedTokenTrainer(**trainer_kwargs)
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    (args.output_dir / "layoutlmv3_field_schema.json").write_text(
        json.dumps({"labels": LABELS, "fields": FIELDS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved LayoutLMv3 field checkpoint to {args.output_dir}")
    print(f"Set LAYOUTLMV3_MODEL_DIR={args.output_dir}")


if __name__ == "__main__":
    main()
