import json
import os
from typing import Iterable, List, Optional, Sequence, Set, Tuple


TOKEN_LIST_KEYS = (
    "tokens",
    "ids",
    "sample_ids",
    "scene_tokens",
    "sample_tokens",
)


def load_token_list(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_tokens = None
    if isinstance(data, list):
        raw_tokens = data
    elif isinstance(data, dict):
        for key in TOKEN_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                raw_tokens = value
                break
        if raw_tokens is None:
            available = ", ".join(sorted(data.keys()))
            raise ValueError(
                f"Could not find a token list in {json_path}. "
                f"Expected one of {TOKEN_LIST_KEYS}; found keys: {available}"
            )
    else:
        raise ValueError(
            f"Unsupported token list format in {json_path}: {type(data).__name__}"
        )

    tokens: List[str] = []
    seen: Set[str] = set()
    for item in raw_tokens:
        if isinstance(item, dict):
            candidate = None
            for key in ("token", "id", "sample_id", "scene_token"):
                if key in item:
                    candidate = item[key]
                    break
            if candidate is None:
                raise ValueError(
                    "Token entries stored as objects must contain one of "
                    "'token', 'id', 'sample_id', or 'scene_token'."
                )
        else:
            candidate = item

        token = str(candidate)
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    return tokens


def get_dataset_tokens(dataset) -> List[str]:
    if hasattr(dataset, "_scene_loader") and hasattr(dataset._scene_loader, "tokens"):
        return list(dataset._scene_loader.tokens)

    if hasattr(dataset, "scenes"):
        return [scene[0] for scene in dataset.scenes]

    raise AttributeError(
        "Dataset does not expose tokens via '_scene_loader.tokens' or 'scenes'."
    )


def resolve_indices_from_tokens(
    dataset_tokens: Sequence[str],
    requested_tokens: Sequence[str],
) -> Tuple[List[int], List[str]]:
    token_to_idx = {token: idx for idx, token in enumerate(dataset_tokens)}

    selected_indices: List[int] = []
    missing_tokens: List[str] = []
    for token in requested_tokens:
        idx = token_to_idx.get(token)
        if idx is None:
            missing_tokens.append(token)
            continue
        selected_indices.append(idx)

    return selected_indices, missing_tokens


def partition_indices(
    indices: Sequence[int],
    sample_num: int,
    num_parts: int,
) -> List[int]:
    if sample_num == 0:
        return list(indices)

    total_len = len(indices)
    part_len = total_len // num_parts
    start_idx = (sample_num - 1) * part_len
    end_idx = sample_num * part_len if sample_num < num_parts else total_len
    return list(indices[start_idx:end_idx])


def collect_processed_tokens(directories: Optional[Iterable[str]]) -> Set[str]:
    processed_tokens: Set[str] = set()
    if not directories:
        return processed_tokens

    seen_dirs: Set[str] = set()
    for directory in directories:
        if not directory:
            continue

        directory = os.path.abspath(directory)
        if directory in seen_dirs or not os.path.isdir(directory):
            continue
        seen_dirs.add(directory)

        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".json"):
                processed_tokens.add(os.path.splitext(entry.name)[0])

    return processed_tokens
