from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fod_annotations.csv"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo_dataset"
    / "images"
    / "test"
)

SAMPLE_DIR = (
    PROJECT_ROOT
    / "examples"
    / "sample_images"
)

# Number of different FOD classes to show in the UI
SAMPLES_PER_CLASS = 1

# Maximum number of samples in the UI
MAX_SAMPLES = 12


def main():

    if not ANNOTATION_CSV.exists():

        raise FileNotFoundError(
            f"Annotation CSV not found:\n{ANNOTATION_CSV}"
        )

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGE_DIR}"
        )

    df = pd.read_csv(
        ANNOTATION_CSV
    )

    # Only use images belonging to the official test split.
    test_df = df[
        df["split"] == "test"
    ].copy()

    # Remove duplicate image/class combinations.
    test_df = test_df[
        [
            "image_id",
            "filename",
            "class_name"
        ]
    ].drop_duplicates()

    # Sort so the selection is deterministic.
    test_df = test_df.sort_values(
        [
            "class_name",
            "image_id"
        ]
    )

    # Start from a clean sample directory.
    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for old_file in SAMPLE_DIR.iterdir():

        if old_file.is_file():

            old_file.unlink()

    selected_rows = []

    # Pick one image for each different FOD class.
    for class_name, class_group in test_df.groupby(
        "class_name",
        sort=True
    ):

        selected = class_group.head(
            SAMPLES_PER_CLASS
        )

        for _, row in selected.iterrows():

            selected_rows.append(
                row
            )

            if len(selected_rows) >= MAX_SAMPLES:
                break

        if len(selected_rows) >= MAX_SAMPLES:
            break

    # Copy selected images.
    copied = 0

    for row in selected_rows:

        filename = str(
            row["filename"]
        )

        source = (
            TEST_IMAGE_DIR
            / filename
        )

        if not source.exists():

            print(
                f"Skipping missing image: {filename}"
            )

            continue

        destination = (
            SAMPLE_DIR
            / filename
        )

        shutil.copy2(
            source,
            destination
        )

        print(
            f"Copied: {filename} "
            f"-> {row['class_name']}"
        )

        copied += 1

    print()
    print(
        f"Sample images created: {copied}"
    )

    print(
        f"Sample directory:\n{SAMPLE_DIR}"
    )


if __name__ == "__main__":
    main()