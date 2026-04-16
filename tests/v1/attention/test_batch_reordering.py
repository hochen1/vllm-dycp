# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import pytest

from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_cp_and_normal,
    reorder_batch_to_split_decodes_and_prefills,
)


class MockInputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed_tokens_cpu

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]
        self.num_computed_tokens_cpu[i], self.num_computed_tokens_cpu[j] = (
            self.num_computed_tokens_cpu[j],
            self.num_computed_tokens_cpu[i],
        )


class MockSchedulerOutput:
    def __init__(self, num_scheduled_tokens=None, cp_rank_scheduled_tokens=None):
        self.num_scheduled_tokens = num_scheduled_tokens
        self.cp_rank_scheduled_tokens = cp_rank_scheduled_tokens


@dataclass
class ReorderTestCase:
    requests: list[tuple[int, int]]  # (num_scheduled_tokens, num_computed_tokens)
    expected_order: list[int]
    expected_modified: bool
    decode_threshold: int = 1


# Test cases for batch reordering
REORDER_TEST_CASES = {
    "all_decodes": ReorderTestCase(
        requests=[(1, 10), (1, 20), (1, 30)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "all_prefills": ReorderTestCase(
        requests=[(100, 100), (200, 200), (300, 300)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "mixed_interleaved": ReorderTestCase(
        requests=[(100, 100), (1, 10), (200, 200), (1, 20)],
        expected_order=[3, 1, 2, 0],  # Only swap 0↔3, keep 1 and 2 in place
        expected_modified=True,
    ),
    "already_ordered": ReorderTestCase(
        requests=[(1, 10), (1, 20), (100, 100), (200, 0)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
    ),
    "single_request": ReorderTestCase(
        requests=[(1, 10)],
        expected_order=[0],
        expected_modified=False,
    ),
    "higher_threshold": ReorderTestCase(
        requests=[(2, 10), (3, 20), (5, 30), (6, 40)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
        decode_threshold=4,
    ),
    "decodes_at_end": ReorderTestCase(
        requests=[(100, 100), (200, 200), (1, 10), (1, 20)],
        expected_order=[2, 3, 0, 1],
        expected_modified=True,
    ),
    "decode_extend_prefill": ReorderTestCase(
        requests=[(100, 0), (10, 50), (1, 10)],
        expected_order=[2, 1, 0],
        expected_modified=True,
    ),
    "extend_prefill_only": ReorderTestCase(
        requests=[(100, 0), (10, 50), (200, 0), (20, 75)],
        expected_order=[3, 1, 2, 0],  # Only swap 0↔3, keep 1 and 2 in place
        expected_modified=True,
    ),
    "complicated_mixed_interleaved": ReorderTestCase(
        requests=[
            (1, 20),
            (1, 50),
            (374, 0),
            (300, 20),
            (1, 20),
            (256, 0),
            (1, 5),
            (27, 0),
            (1, 4),
        ],
        expected_order=[0, 1, 6, 8, 4, 3, 2, 7, 5],
        expected_modified=True,
    ),
}


@pytest.mark.parametrize(
    "test_case", REORDER_TEST_CASES.values(), ids=REORDER_TEST_CASES.keys()
)
def test_reorder_batch_to_split_decodes_and_prefills(test_case: ReorderTestCase):
    req_ids = [f"r{i}" for i in range(len(test_case.requests))]
    num_computed_tokens = np.array([r[1] for r in test_case.requests], dtype=np.int32)
    num_scheduled_tokens = {f"r{i}": r[0] for i, r in enumerate(test_case.requests)}

    input_batch = MockInputBatch(req_ids, num_computed_tokens)
    scheduler_output = MockSchedulerOutput(num_scheduled_tokens)

    modified = reorder_batch_to_split_decodes_and_prefills(
        input_batch, scheduler_output, decode_threshold=test_case.decode_threshold
    )

    expected_req_ids = [f"r{i}" for i in test_case.expected_order]

    assert modified == test_case.expected_modified, (
        f"Expected modified={test_case.expected_modified}, got {modified}"
    )
    assert input_batch.req_ids == expected_req_ids, (
        f"Expected order {expected_req_ids}, got {input_batch.req_ids}"
    )


@dataclass
class CPReorderTestCase:
    cp_sizes: list[int]
    payloads: list[int]
    expected_order: list[int]
    expected_modified: bool


CP_REORDER_TEST_CASES = {
    "already_ordered": CPReorderTestCase(
        cp_sizes=[2, 2, 1, 1],
        payloads=[10, 20, 30, 40],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
    ),
    "stable_partition_interleaved": CPReorderTestCase(
        cp_sizes=[1, 2, 1, 3, 2, 1],
        payloads=[10, 20, 30, 40, 50, 60],
        expected_order=[1, 3, 4, 0, 2, 5],
        expected_modified=True,
    ),
    "cp_and_ncp_relative_order_kept": CPReorderTestCase(
        cp_sizes=[3, 1, 2, 1, 1, 4, 2],
        payloads=[101, 102, 103, 104, 105, 106, 107],
        expected_order=[0, 2, 5, 6, 1, 3, 4],
        expected_modified=True,
    ),
}


@pytest.mark.parametrize(
    "test_case", CP_REORDER_TEST_CASES.values(), ids=CP_REORDER_TEST_CASES.keys()
)
def test_reorder_batch_to_split_cp_and_normal(test_case: CPReorderTestCase):
    req_ids = [f"r{i}" for i in range(len(test_case.cp_sizes))]
    num_computed_tokens = np.array(test_case.payloads, dtype=np.int32)
    cp_rank_scheduled_tokens = {
        f"r{i}": cp_size for i, cp_size in enumerate(test_case.cp_sizes)
    }

    input_batch = MockInputBatch(req_ids, num_computed_tokens)
    scheduler_output = MockSchedulerOutput(
        cp_rank_scheduled_tokens=cp_rank_scheduled_tokens
    )

    modified = reorder_batch_to_split_cp_and_normal(
        input_batch, scheduler_output
    )

    expected_req_ids = [f"r{i}" for i in test_case.expected_order]
    expected_payloads = np.array(
        [test_case.payloads[i] for i in test_case.expected_order],
        dtype=np.int32,
    )

    assert modified == test_case.expected_modified, (
        f"Expected modified={test_case.expected_modified}, got {modified}"
    )
    assert input_batch.req_ids == expected_req_ids, (
        f"Expected order {expected_req_ids}, got {input_batch.req_ids}"
    )
    assert np.array_equal(input_batch.num_computed_tokens_cpu, expected_payloads), (
        f"Expected payload order {expected_payloads.tolist()}, "
        f"got {input_batch.num_computed_tokens_cpu.tolist()}"
    )
