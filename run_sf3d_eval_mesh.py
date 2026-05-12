#!/usr/bin/env python3
from sf3d_eval_lib_mesh_metric.config import parse_args
from sf3d_eval_lib_mesh_metric.runner import run_eval


def main() -> None:
    cfg = parse_args()
    run_eval(cfg)


if __name__ == "__main__":
    main()
