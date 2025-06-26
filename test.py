import sys

sys.path.insert(0, ".")

from wanvideo import WanPipeline
from wanvideo.utils import debug_logger

with debug_logger() as logger:
    pipeline = WanPipeline.from_original_t2v_pretrained(device="cuda", dtype="bfloat16")
    print(f"{pipeline=}")
    result = pipeline(
        prompt="A beautiful landscape with mountains and a river",
        num_inference_steps=2,
    )
    print(f"{result=}")
