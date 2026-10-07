from pathlib import Path
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_processed_annotations_exist():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "fod_annotations.csv"
    )

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) > 0
    assert "image_id" in df.columns
    assert "filename" in df.columns
    assert "class_name" in df.columns


def test_yolo_dataset_exists():
    dataset_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "yolo_dataset"
    )

    assert (
        dataset_dir
        / "images"
        / "train"
    ).exists()

    assert (
        dataset_dir
        / "images"
        / "val"
    ).exists()

    assert (
        dataset_dir
        / "images"
        / "test"
    ).exists()

    assert (
        dataset_dir
        / "labels"
        / "train"
    ).exists()


def test_model_exists():
    model_path = (
        PROJECT_ROOT
        / "models"
        / "fod_yolo_baseline_best.pt"
    )

    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_data_yaml_exists():
    yaml_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "yolo_dataset"
        / "data.yaml"
    )

    assert yaml_path.exists()

    config = yaml.safe_load(
        yaml_path.read_text(
            encoding="utf-8"
        )
    )

    assert "train" in config
    assert "val" in config
    assert "test" in config
    assert "nc" in config
    assert "names" in config
