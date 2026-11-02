from loguru import logger
import os
from typing import Dict


def compute_score_remote(problem: Dict, completion: str):
    import requests

    keck_base_url = os.environ.get("KECK_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    solution_str = completion
    ground_truth = problem["ground_truth"]["pytorch_module"]
    response = requests.post(
        url=f"{keck_base_url}/compute_score",
        data={
            "cuda_code": solution_str,
            "torch_code": ground_truth,
        },
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    res_json = response.json()
    logger.info(res_json)
    return res_json
