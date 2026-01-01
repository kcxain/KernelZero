import json


def load_json(file_path: str):
    """
    Support loading JSON, JSONL, and Parquet files.
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif file_path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(file_path).to_dict(orient="records")
    else:
        raise ValueError("Unsupported file format: {}".format(file_path))


def save_json(data, file_path: str):
    """
    Support saving JSON, JSONL, and Parquet files.
    """
    if file_path.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    elif file_path.endswith(".jsonl"):
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif file_path.endswith(".parquet"):
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError("Unsupported file format: {}".format(file_path))
