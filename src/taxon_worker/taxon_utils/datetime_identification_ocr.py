import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def is_correct_datetime_format(date):
    """Check if the date is in the correct format."""

    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d %H:%M:%S")
    if date is None:
        return False

    try:
        date = pd.to_datetime(date, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    except ValueError:
        date = None

    if date is None or pd.isnull(date):
        return False
    return True


def normalize_tokens(tokens):
    """Normalize the tokens."""
    text = " ".join(tokens)

    replacements = {
        "*": ":",
        "O": "0",
        "o": "0",
        "l": "1",
        "I": "1",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # fix weird spacing inside numbers
    text = re.sub(r"(\d)\s+(\d)", r"\1:\2", text)

    return text


def extract_datetime(text):
    """Extract the datetime from the text."""
    patterns = [
        r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4}).*?(\d{1,2})[:\.](\d{2})\s*(AM|PM)?",
        r"(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2}).*?(\d{1,2})[:\.](\d{2})[:\.](\d{2}).*?(AM|PM)?",
        r"(\d{1,2})[\.](\d{1,2})[\.](\d{4}).*?(\d{2})[:\.](\d{2})[:\.](\d{2})",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.groups()

    return None


def validate_parsed_datetime(value):
    """Validate the parsed datetime."""
    if value is None:
        return None

    if value > datetime.now():
        return None

    return value


def parse_candidate(groups):
    """Parse the candidate datetime."""
    try:
        # format: MM/DD/YYYY HH:MM AM/PM
        if len(groups) == 6:
            a, b, y, h, mi, ampm = groups

            # heuristic: detect format
            # if first > 12 -> likely day/month swapped
            a, b = int(a), int(b)
            y = int(y)
            h, mi = int(h), int(mi)

            if y < 100:
                y += 2000

            # assume MM/DD/YYYY (Cuddeback default)
            month, day = a, b

            if ampm:
                ampm = ampm.upper()
                if ampm == "PM" and h != 12:
                    h += 12
                if ampm == "AM" and h == 12:
                    h = 0

            return validate_parsed_datetime(datetime(y, month, day, h, mi))

        # format: YYYY-MM-DD HH:MM:SS
        elif len(groups) == 7:
            y, m, d, h, mi, s, ampm = groups

            y, m, d = int(y), int(m), int(d)
            h, mi, s = int(h), int(mi), int(s)

            if ampm:
                ampm = ampm.upper()
                if ampm == "PM" and h != 12:
                    h += 12
                if ampm == "AM" and h == 12:
                    h = 0

            return validate_parsed_datetime(datetime(y, m, d, h, mi, s))

    except Exception as e:
        print(e)
        return None


def parse_iso(tokens):
    """Parse the ISO datetime."""

    def safe_int(x):
        x = re.sub(r"[^0-9]", "", x)  # keep digits only
        return int(x) if x else 0

    for i in range(len(tokens) - 1):
        if re.match(r"\d{4}-\d{1,2}-\d{1,2}", tokens[i]):
            date_part = tokens[i]
            time_part = tokens[i + 1]

            y, m, d = map(int, date_part.split("-"))

            time_part = time_part.replace("*", ":").replace(".", ":")

            parts = time_part.split(":") + ["0", "0", "0"]

            h = safe_int(parts[0])
            mi = safe_int(parts[1])
            s = safe_int(parts[2])

            return validate_parsed_datetime(datetime(y, m, d, h, mi, s))

    return None


def extract_best_datetime(tokens):
    """Extract the best datetime from the tokens."""
    # 1. TRY ISO FIRST
    try:
        iso = parse_iso(tokens)
    except Exception as e:
        print(e)
        iso = None

    if iso:
        return iso

    # 2. fallback to regex
    text = normalize_tokens(tokens)
    groups = extract_datetime(text)

    if not groups:
        return None

    return parse_candidate(groups)


def process_datetime_from_ocr(detections):
    """Process the datetime from the OCR detections."""
    results = []

    for tokens in detections:
        dt = extract_best_datetime(tokens)
        results.append({"input": tokens, "datetime": dt})

    return results


class OCRDataset(Dataset):
    """OCR dataset."""

    def __init__(
        self,
        # metadata: pd.DataFrame,
        # root_path: str,
        # path_col: str = "_image_url",
        image_paths,
        image_shape: tuple[int, int] = (720, 1920),
    ):
        # self.metadata = metadata
        # self.root_path = root_path
        # self.path_col = path_col
        self.image_paths = image_paths
        self.image_shape = image_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # row = self.metadata.iloc[idx]
        # image_path = Path(os.path.join(self.root_path, row[self.path_col]))
        image_path = Path(os.path.join(self.image_paths[idx]))
        if image_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            image_bgr = cv2.imread(str(image_path))
        else:
            # read video frame
            cap = cv2.VideoCapture(str(image_path))
            ret, image_bgr = cap.read()
            cap.release()

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.image_shape is not None:
            image_gray = cv2.resize(image_gray, self.image_shape)

        return image_gray


def _process_datetime_without_spaces(text_raw, results):
    """Process the datetime without spaces."""
    _idxs = []
    _text_raw = []
    for idx, res in enumerate(results):
        if res is None:
            _idxs.append(idx)
            _text_raw.append([t.replace(" ", "") for t in text_raw[idx]])

    if _idxs:
        _results = [r["datetime"] for r in process_datetime_from_ocr(_text_raw)]
        for idx, res in zip(_idxs, _results):
            if res is not None:
                results[idx] = res
    return results


def get_datetime_from_ocr(
    metadata: pd.DataFrame,
    root_path: str,
    path_col: str = "_image_url",
    batch_size: int = 1,
    num_workers: int = 8,
    image_shape: tuple[int, int] = None,
) -> tuple[list, list]:
    """Get the datetime from the OCR."""
    # create dataset
    dataset = OCRDataset(metadata, root_path, path_col, image_shape)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
        prefetch_factor=8,
    )

    # create ocr reader
    reader = easyocr.Reader(["en"])

    results_raw = []
    results_clean = []
    for batch in tqdm(dataloader):
        # read text from image
        text_raw = reader.readtext_batched(batch, detail=0)
        # process text to datetime
        results = [r["datetime"] for r in process_datetime_from_ocr(text_raw)]

        # backup - run processing again but remove spaces before processing
        results = _process_datetime_without_spaces(text_raw, results)

        # save results
        results_raw.extend(text_raw)
        results_clean.extend(results)

    return results_raw, results_clean
