python ./gen_json.py \
  --avg_input_len 4096 \
  --avg_output_len 1024 \
  --max_input_len 262144 \
  --max_output_len 1024 \
  --gap_len 128 \
  --num_requests 1280 \
  --out trace_256k.json
