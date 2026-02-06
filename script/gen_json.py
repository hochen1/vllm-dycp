#!/usr/bin/env python3
"""
gen_trace.py
生成请求长度 trace 的 JSON 文件。

用法：
  python gen_trace.py \
      --avg_input_len  512 \
      --avg_output_len 256 \
      --max_input_len  2048 \
      --max_output_len 1024 \
      --gap_len        100 \
      --num_requests   1000 \
      --out            trace.json
"""

import argparse
import json
import random
import sys

def build_trace(args):
    """
    返回 List[[int, int]]，格式为 [input_len, output_len]
    每 gap_len 个请求里随机挑 1 个设为 max，其余为 avg。
    """
    trace = []
    for i in range(args.num_requests):
        # 决定是否在本 gap 段里放 max
        if i % args.gap_len == 0:          # 每个 gap 段的起始
            pick_one = random.randint(0, args.gap_len - 1)
        # 当前段内偏移
        offset = i % args.gap_len
        if offset == pick_one:
            inp = args.max_input_len
            out = args.max_output_len
        else:
            inp = args.avg_input_len
            out = args.avg_output_len
        trace.append([inp, out])
    return trace

def main():
    parser = argparse.ArgumentParser(description="Generate request length trace JSON.")
    parser.add_argument("--avg_input_len",  type=int, required=True)
    parser.add_argument("--avg_output_len", type=int, required=True)
    parser.add_argument("--max_input_len",  type=int, required=True)
    parser.add_argument("--max_output_len", type=int, required=True)
    parser.add_argument("--gap_len",        type=int, required=True)
    parser.add_argument("--num_requests",   type=int, required=True)
    parser.add_argument("--out",            type=str, default="trace.json")
    args = parser.parse_args()

    if args.gap_len <= 0 or args.num_requests <= 0:
        print("gap_len 和 num_requests 必须为正整数", file=sys.stderr)
        sys.exit(1)

    trace = build_trace(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"已生成 {args.num_requests} 条记录 -> {args.out}")

if __name__ == "__main__":
    main()
