from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import pandas as pd


class FODDataIngestion:
    """
    Reads the FOD-A Pascal VOC dataset and converts image-level XML
    annotations + weather/light metadata into a structured DataFrame.

    One row represents one annotated FOD object.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

        self.voc_dir = (
            self.data_root
            / "FODPascalVOCFormat-V.2.1"
            / "VOC2007"
        )

        self.annotations_dir = self.voc_dir / "Annotations"
        self.images_dir = self.voc_dir / "JPEGImages"
        self.image_sets_dir = self.voc_dir / "ImageSets" / "Main"
        self.categorization_dir = (
            self.image_sets_dir / "CategorizationData"
        )

        self.trainval_file = self.image_sets_dir / "trainval.txt"
        self.test_file = self.image_sets_dir / "test.txt"

        self.categorization_file = (
            self.categorization_dir
            / "FOD_categorization_annotations.csv"
        )

    def validate_paths(self) -> None:
        """Validate required dataset paths before ingestion."""

        required_paths = {
            "VOC directory": self.voc_dir,
            "Annotations directory": self.annotations_dir,
            "JPEGImages directory": self.images_dir,
            "ImageSets directory": self.image_sets_dir,
            "trainval.txt": self.trainval_file,
            "test.txt": self.test_file,
            "Categorization CSV": self.categorization_file,
        }

        missing_paths = [
            name
            for name, path in required_paths.items()
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Missing required dataset components: "
                + ", ".join(missing_paths)
            )

    @staticmethod
    def _read_split_file(file_path: Path) -> set:
        """
        Read image IDs from a Pascal VOC split file.

        Example:
        000000
        000001
        """
        with file_path.open("r", encoding="utf-8") as file:
            return {
                line.strip()
                for line in file
                if line.strip()
            }

    def _load_split_mapping(self) -> Dict[str, str]:
        """
        Create a mapping:
        image_id -> dataset split
        """

        trainval_ids = self._read_split_file(self.trainval_file)
        test_ids = self._read_split_file(self.test_file)

        overlap = trainval_ids.intersection(test_ids)

        if overlap:
            print(
                f"Warning: Found {len(overlap)} image IDs in both "
                "trainval and test. Assigning them to test split."
            )

        trainval_ids = trainval_ids - overlap

        split_mapping = {
            image_id: "trainval"
            for image_id in trainval_ids
        }

        split_mapping.update(
            {
                image_id: "test"
                for image_id in test_ids
            }
        )

        return split_mapping

    def _load_environment_metadata(self) -> pd.DataFrame:
        """
        Load weather and light metadata.

        Raw dataset encoding:
            Weather: 0 = Dry, 1 = Wet
            Light:   0 = Bright, 1 = Dim, 2 = Dark
        """

        df = pd.read_csv(self.categorization_file)

        required_columns = {"File", "Weather", "Light"}

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                "Missing columns in categorization CSV: "
                + ", ".join(sorted(missing_columns))
            )

        df["image_id"] = (
            df["File"]
            .astype(str)
            .str.replace(".jpg", "", regex=False)
        )

        weather_mapping = {
            0: "Dry",
            1: "Wet",
        }

        light_mapping = {
            0: "Bright",
            1: "Dim",
            2: "Dark",
        }

        df["weather_label"] = df["Weather"].map(weather_mapping)
        df["light_label"] = df["Light"].map(light_mapping)

        unknown_weather = df.loc[
            df["weather_label"].isna(), "Weather"
        ].dropna().unique()

        unknown_light = df.loc[
            df["light_label"].isna(), "Light"
        ].dropna().unique()

        if len(unknown_weather) > 0:
            raise ValueError(
                f"Unknown weather labels found: {unknown_weather.tolist()}"
            )

        if len(unknown_light) > 0:
            raise ValueError(
                f"Unknown light labels found: {unknown_light.tolist()}"
            )

        return df[
            [
                "image_id",
                "weather_label",
                "light_label",
            ]
        ]

    @staticmethod
    def _extract_attributes(object_element: ET.Element) -> Dict[str, str]:
        """
        Extract optional annotation attributes such as:
            track_id
            keyframe
        """

        attributes = {}

        attributes_element = object_element.find("attributes")

        if attributes_element is None:
            return attributes

        for attribute in attributes_element.findall("attribute"):
            name = attribute.findtext("name")
            value = attribute.findtext("value")

            if name:
                attributes[name] = value

        return attributes

    def _parse_annotation(
        self,
        xml_path: Path,
        split_mapping: Dict[str, str],
    ) -> List[Dict]:
        """
        Parse one Pascal VOC XML file.

        Returns one dictionary per FOD object.
        """

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise ValueError(
                f"Invalid XML annotation: {xml_path}"
            ) from exc

        root = tree.getroot()

        filename = root.findtext("filename")

        if not filename:
            raise ValueError(
                f"Missing filename in annotation: {xml_path}"
            )

        image_id = Path(filename).stem

        size_element = root.find("size")

        if size_element is None:
            raise ValueError(
                f"Missing image size in annotation: {xml_path}"
            )

        width = int(float(size_element.findtext("width")))
        height = int(float(size_element.findtext("height")))

        split = split_mapping.get(image_id)

        if split is None:
            split = "unassigned"

        objects = root.findall("object")

        records = []

        for object_element in objects:
            class_name = object_element.findtext("name")

            if not class_name:
                continue

            bbox_element = object_element.find("bndbox")

            if bbox_element is None:
                continue

            xmin = float(bbox_element.findtext("xmin"))
            ymin = float(bbox_element.findtext("ymin"))
            xmax = float(bbox_element.findtext("xmax"))
            ymax = float(bbox_element.findtext("ymax"))

            # Basic bounding-box validation
            if xmin < 0 or ymin < 0:
                raise ValueError(
                    f"Negative bounding box found in {xml_path}"
                )

            if xmax > width or ymax > height:
                raise ValueError(
                    f"Bounding box exceeds image boundaries in {xml_path}"
                )

            if xmin >= xmax or ymin >= ymax:
                raise ValueError(
                    f"Invalid bounding box in {xml_path}"
                )

            attributes = self._extract_attributes(object_element)

            record = {
                "image_id": image_id,
                "filename": filename,
                "image_path": str(
                    self.images_dir / filename
                ),
                "annotation_path": str(xml_path),
                "split": split,
                "width": width,
                "height": height,
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "occluded": object_element.findtext("occluded"),
                "truncated": object_element.findtext("truncated"),
                "difficult": object_element.findtext("difficult"),
                "pose": object_element.findtext("pose"),
                "track_id": attributes.get("track_id"),
                "keyframe": attributes.get("keyframe"),
            }

            records.append(record)

        return records

    def build_dataset(self) -> pd.DataFrame:
        """
        Build the complete structured annotation DataFrame.
        """

        self.validate_paths()

        split_mapping = self._load_split_mapping()

        environment_df = self._load_environment_metadata()

        annotation_files = sorted(
            self.annotations_dir.glob("*.xml")
        )

        if not annotation_files:
            raise FileNotFoundError(
                "No XML annotation files found."
            )

        records = []

        print(
            f"Found {len(annotation_files):,} annotation files."
        )

        for index, xml_path in enumerate(annotation_files, start=1):
            parsed_records = self._parse_annotation(
                xml_path,
                split_mapping,
            )

            records.extend(parsed_records)

            if index % 1000 == 0:
                print(
                    f"Processed {index:,}/{len(annotation_files):,} "
                    "annotations..."
                )

        annotation_df = pd.DataFrame(records)

        if annotation_df.empty:
            raise ValueError(
                "No valid object annotations were extracted."
            )

        dataset_df = annotation_df.merge(
            environment_df,
            on="image_id",
            how="left",
            validate="many_to_one",
        )

        missing_environment = dataset_df[
            dataset_df["weather_label"].isna()
            | dataset_df["light_label"].isna()
        ]

        if not missing_environment.empty:
            raise ValueError(
                "Weather/Light metadata missing for "
                f"{len(missing_environment):,} annotation rows."
            )

        dataset_df["image_exists"] = dataset_df[
            "image_path"
        ].map(
            lambda path: Path(path).exists()
        )

        missing_images = dataset_df[
            ~dataset_df["image_exists"]
        ]

        if not missing_images.empty:
            raise FileNotFoundError(
                f"{len(missing_images):,} images referenced by "
                "annotations do not exist."
            )

        # Add normalized bounding-box coordinates.
        dataset_df["x_center"] = (
            (dataset_df["xmin"] + dataset_df["xmax"]) / 2
        ) / dataset_df["width"]

        dataset_df["y_center"] = (
            (dataset_df["ymin"] + dataset_df["ymax"]) / 2
        ) / dataset_df["height"]

        dataset_df["box_width"] = (
            dataset_df["xmax"] - dataset_df["xmin"]
        ) / dataset_df["width"]

        dataset_df["box_height"] = (
            dataset_df["ymax"] - dataset_df["ymin"]
        ) / dataset_df["height"]
        unassigned_count = (
            dataset_df["split"]
            .eq("unassigned")
            .sum()
        )

        if unassigned_count > 0:
            print(
                f"Warning: {unassigned_count:,} annotation rows "
                "belong to images without a trainval/test split."
            )
        return dataset_df


def main() -> None:
    """Run the ingestion process and save the processed metadata."""

    project_root = Path(__file__).resolve().parents[2]

    data_root = project_root / "data" / "raw"

    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "fod_annotations.csv"

    ingestion = FODDataIngestion(
        data_root=str(data_root)
    )

    dataset_df = ingestion.build_dataset()

    dataset_df.to_csv(
        output_file,
        index=False,
    )

    print("\nData ingestion completed successfully.")
    print(f"Rows: {len(dataset_df):,}")
    print(
        f"Unique images: "
        f"{dataset_df['image_id'].nunique():,}"
    )
    print(
        f"Unique classes: "
        f"{dataset_df['class_name'].nunique():,}"
    )
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()