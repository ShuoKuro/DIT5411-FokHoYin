import argparse
from pathlib import Path
from zipfile import ZipFile


def extract_dataset(archive_path, output_dir, metadata_encoding="big5"):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, metadata_encoding=metadata_encoding) as archive:
        members = archive.infolist()
        for member in members:
            destination = (output_dir / member.filename).resolve()
            try:
                destination.relative_to(output_dir)
            except ValueError as error:
                raise ValueError(f"Unsafe archive member path: {member.filename}") from error
        archive.extractall(output_dir, members=members)

    cleaned_data = output_dir / "cleaned_data"
    if not cleaned_data.is_dir():
        raise RuntimeError(f"Extraction finished but the expected directory is missing: {cleaned_data}")
    print(f"Extracted {len(members)} archive entries to {output_dir}")
    print(f"Dataset directory: {cleaned_data}")
    return cleaned_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract the official dataset with Chinese filename support on macOS."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--metadata-encoding", default="big5")
    args = parser.parse_args()
    extract_dataset(args.archive, args.output_dir, args.metadata_encoding)


if __name__ == "__main__":
    main()
