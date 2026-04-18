python ./gen_json.py \
  --avg_input_len 4096 \
  --avg_output_len 1 \
  --max_input_len 524288 \
  --max_output_len 1 \
  --gap_len 100 \
  --num_requests 4000 \
  --out temp.json
