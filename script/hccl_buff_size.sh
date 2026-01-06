#!/bin/bash
# refer to https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/aolapi/context/aclnnMoeDistributeDispatchV2.md

get_json_int_from_file_by_key() {
    local key="$1"
    local file="$2"
    local value=$(grep -o "\"${key}\"[[:space:]]*:[[:space:]]*[0-9]\+" "$file" | head -n1 | sed 's/.*:[[:space:]]*\([0-9]\+\)/\1/')
    if [[ -z "$value" ]]; then
        echo "Error: Key '$key' not found or invalid in $file" >&2
        return 1
    fi
    echo "$value"
}

get_json_string_from_file_by_key() {
    local key="$1"
    local file="$2"
        local value=$(grep -o "\"${key}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$file" | head -n1 | sed 's/.*:[[:space:]]*"\([^"]*\)"/\1/')
    
    if [[ -z "$value" ]]; then
        echo "Error: String key '$key' not found or invalid in $file" >&2
        return 1
    fi
    echo "$value"
}

align32() { echo $(( (($1 + 31) / 32) * 32 )); }
align512() { echo $(( (($1 + 511) / 512) * 512 )); }

calculate_hccl_buffsize() {
    local model_path="$1"
    local env_type="$2"
    local data_parallel_size="$3"
    local tensor_parallel_size="$4"
    local max_num_seqs="$5"
    
    if [[ -z "$model_path" ]] ||
       [[ -z "$env_type" ]] ||
       [[ -z "$data_parallel_size" ]] ||
       [[ -z "$tensor_parallel_size" ]] ||
       [[ -z "$max_num_seqs" ]]; then
        echo "Error: Missing required arguments in calculate_hccl_buffsize" >&2
        echo "Usage: calculate_hccl_buffsize <model_path> <env_type> <data_parallel_size> <tensor_parallel_size> <max_num_seqs>" >&2
        return 1
    fi
    
    local config_file="$model_path/config.json"
    if [[ ! -f "$config_file" ]]; then
        echo "Error: $config_file not found" >&2
        return 1
    fi
    
    hidden_size=$(get_json_int_from_file_by_key "hidden_size" "$config_file") || return 1
    num_experts_per_tok=$(get_json_int_from_file_by_key "num_experts_per_tok" "$config_file") || return 1
    n_routed_experts=$(get_json_int_from_file_by_key "n_routed_experts" "$config_file") || return 1
    n_shared_experts=$(get_json_int_from_file_by_key "n_shared_experts" "$config_file") || return 1
    torch_dtype=$(get_json_string_from_file_by_key "torch_dtype" "$config_file") || return 1
    
    case "$torch_dtype" in
        *"float16"*|*"fp16"*)
            dtype_bytes=2
            ;;
        *"bfloat16"*|*"bf16"*)
            dtype_bytes=2
            ;;
        *"float32"*|*"fp32"*)
            dtype_bytes=4
            ;;
        *)
            echo "Warning: Unknown dtype '$torch_dtype', using 2 bytes" >&2
            dtype_bytes=2
            ;;
    esac
    
    ep_world_size=$((data_parallel_size * tensor_parallel_size))
    if [[ $ep_world_size -eq 0 ]]; then
        echo "Error: ep_world_size is zero" >&2
        return 1
    fi
    local_expert_num=$((n_routed_experts / ep_world_size))
    
    H=$hidden_size
    K=$num_experts_per_tok
    moeExpertNum=$n_routed_experts
    sharedExpertNum=$n_shared_experts
    localExpertNum=$local_expert_num
    maxBs=$max_num_seqs
    Bs=$max_num_seqs
    
    case "$env_type" in
        a2)
            # moeExpertNum * Bs * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB
            k_aligned=$(( ((K + 7) / 8) * 8 ))
            uint32_bytes=4

            inner_part=$(( H * dtype_bytes + 4 * k_aligned * uint32_bytes ))
            extra_bytes=$(( (4 + 100) * 1024 * 1024 ))  # 104 MB → bytes

            total_bytes=$(( moeExpertNum * Bs * inner_part + extra_bytes ))
            mb_size=$((1024 * 1024))
            result_mb=$(( (total_bytes + mb_size - 1) / mb_size ))
            echo "$result_mb"
            ;;
            
        a3)
            # 2 * (localExpertNum * maxBs * epWorldSize * Align512(Align32(2 * H) + 64) + (K + sharedExpertNum) * maxBs * Align512(2 * H))
            # Align512(x) = ((x + 512 - 1) / 512) * 512
            # Align32(x) = ((x + 32 - 1) / 32) * 32
            temp1=$(align32 $((2 * H)))
            temp1=$((temp1 + 64))
            aligned1=$(align512 $temp1)
            
            aligned2=$(align512 $((2 * H)))
            
            part1=$(( localExpertNum * maxBs * ep_world_size * aligned1 ))
            part2=$(( (K + sharedExpertNum) * maxBs * aligned2 ))
            
            total_bytes=$(( 2 * (part1 + part2) ))

            mb_size=$((1024 * 1024))
            result_mb=$(( (total_bytes + mb_size - 1) / mb_size ))

            if [[ $result_mb -gt 2 ]]; then
                echo "$result_mb"
            else
                echo "2"
            fi
            ;;
            
        *)
            echo "Error: Unsupported ENV='$env_type'. Use 'a2' or 'a3'" >&2
            return 1
            ;;
    esac
}

