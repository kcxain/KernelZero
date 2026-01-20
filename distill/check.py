from loguru import logger
import os
from typing import Dict


def compute_score_remote(problem: Dict, completion: str):
    import requests

    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")
    gpu_port = os.environ.get("CUDA_SERVER_PORT", "8000")
    solution_str = completion
    ground_truth = problem["ground_truth"]["pytorch_module"]
    # curl -X POST http://10.200.198.14:8000/compute_score   -F "code=123"   -F "timeout_sec=60"   -F "nvcc_flags=-O2 -arch=sm_80"
    response = requests.post(
        url=f"http://{gpu_ip}:{gpu_port}/compute_score",
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
