import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDeterministicGDNPrefill(unittest.TestCase):
    def scheduler(self, alignment, chunk=1000, page=1, dllm=None):
        scheduler = Scheduler.__new__(Scheduler)
        backend = SimpleNamespace(deterministic_prefill_chunk_alignment=alignment)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend)
        )
        scheduler.chunked_prefill_size = chunk
        scheduler.page_size = page
        scheduler.dllm_config = dllm
        return scheduler

    def test_runtime_scope_and_existing_backend_constraints(self):
        for deterministic, alignment, prefill, decode, expected in (
            (True, 64, "fa3", "triton", 64),
            (True, None, "fa3", "triton", None),
            (True, 64, "flashinfer", "fa3", 4096),
            (True, 64, "triton", "fa3", 4096),
            (True, None, "triton", "fa3", 4096),
            (False, 64, "fa3", "triton", None),
        ):
            with self.subTest(
                deterministic=deterministic,
                alignment=alignment,
                prefill=prefill,
                decode=decode,
            ):
                scheduler = self.scheduler(alignment, chunk=8192)
                with (
                    get_context().override_server_args(
                        enable_deterministic_inference=deterministic,
                        prefill_attention_backend=prefill,
                        decode_attention_backend=decode,
                    ),
                    patch.dict(
                        "os.environ",
                        {
                            "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE": "4096",
                            "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": "4096",
                        },
                    ),
                ):
                    scheduler.init_deterministic_inference_config()
                self.assertEqual(scheduler.truncation_align_size, expected)
                self.assertEqual(
                    scheduler.align_chunked_prefill,
                    deterministic and alignment is not None,
                )

    def test_alignment_lcm_and_permanent_small_budget(self):
        with (
            get_context().override_server_args(
                enable_deterministic_inference=True,
                attention_backend="triton",
                prefill_attention_backend=None,
                decode_attention_backend=None,
            ),
            patch.dict(
                "os.environ", {"SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": "96"}
            ),
        ):
            scheduler = self.scheduler(64)
            scheduler.init_deterministic_inference_config()
            self.assertEqual(scheduler.truncation_align_size, 192)

        with (
            get_context().override_server_args(
                enable_deterministic_inference=True,
                attention_backend="fa3",
                prefill_attention_backend=None,
                decode_attention_backend=None,
            ),
            patch(
                "sglang.srt.managers.scheduler._use_exact_chunk_fill",
                return_value=False,
            ),
        ):
            for chunk, page in ((63, 1), (127, 128)):
                with (
                    self.subTest(chunk=chunk, page=page),
                    self.assertRaises(ValueError),
                ):
                    self.scheduler(
                        64, chunk=chunk, page=page
                    ).init_deterministic_inference_config()
            for chunk in (None, 64, 1000):
                self.scheduler(64, chunk=chunk).init_deterministic_inference_config()
            scheduler = self.scheduler(64, chunk=31, dllm=object())
            scheduler.init_deterministic_inference_config()
            self.assertFalse(scheduler.align_chunked_prefill)
            self.assertIsNone(scheduler.truncation_align_size)

    def test_wrappers_follow_prefill_linear_kernel(self):
        self.assertIsNone(AttentionBackend.deterministic_prefill_chunk_alignment)
        linear = SimpleNamespace(deterministic_prefill_chunk_alignment=64)
        hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid.linear_attn_backend = linear
        hybrid.full_attn_backend = SimpleNamespace(
            deterministic_prefill_chunk_alignment=None
        )
        split = HybridAttnBackend.__new__(HybridAttnBackend)
        split.prefill_backend = hybrid
        split.decode_backend = SimpleNamespace(
            deterministic_prefill_chunk_alignment=4096
        )
        self.assertEqual(split.deterministic_prefill_chunk_alignment, 64)
        linear.deterministic_prefill_chunk_alignment = None
        self.assertIsNone(split.deterministic_prefill_chunk_alignment)

    def parked_continuation_scheduler(self):
        scheduler = MagicMock(spec=Scheduler)
        for name in (
            "enable_priority_preemption",
            "is_hybrid_swa",
            "is_mixed_chunk",
            "enable_lora",
            "enable_hicache_storage",
            "enable_hierarchical_cache",
            "enable_unified_cache_external_linker",
            "enable_overlap",
            "enable_priority_scheduling",
        ):
            setattr(scheduler, name, False)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.min_free_slots_delayer = None
        scheduler.dynamic_chunk_sizer = None
        scheduler.chunked_prefill_size = 1000
        scheduler.align_chunked_prefill = True
        scheduler.truncation_align_size = 64
        scheduler.page_size = 1
        scheduler.max_prefill_bs = 8
        scheduler.max_prefill_tokens = 8192
        scheduler.max_running_requests = 8
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.dllm_config = None
        scheduler.disaggregation_mode = "null"
        scheduler.processed_tokens_counter = 0
        scheduler.policy = MagicMock()
        scheduler.get_num_allocatable_reqs.return_value = 8
        scheduler.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.spec_algorithm = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=AttentionBackend(),
                prefill_aware_swa=False,
            )
        )
        parked = MagicMock(inflight_middle_chunks=0)
        admitted = MagicMock(beam_group=None)
        scheduler.chunked_req = parked
        scheduler.waiting_queue = [admitted]
        running = MagicMock(reqs=[], batch_is_full=False)
        adder = MagicMock(
            can_run_list=[admitted], preempt_list=[], new_chunked_req=None
        )
        adder.add_chunked_req.return_value = parked
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        return scheduler, parked, running, adder

    def test_parked_continuation_is_not_counted_in_another_requests_batch(self):
        scheduler, parked, running, adder = self.parked_continuation_scheduler()
        with (
            get_context().override_server_args(),
            patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder),
            patch("sglang.srt.managers.scheduler.ScheduleBatch.init_new") as init_batch,
            patch("sglang.srt.managers.scheduler.PrefillStats.from_adder"),
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(scheduler, None, running)
        self.assertIs(scheduler.chunked_req, parked)
        self.assertEqual(parked.inflight_middle_chunks, 0)
        self.assertIsNone(init_batch.call_args.kwargs["chunked_req"])
        self.assertTrue(batch.contains_last_prefill_chunk)
        scheduler.load_inquirer._get_num_pending_tokens.assert_called_once_with(
            chunk_deduct=0
        )

    def test_dynamic_prediction_cannot_permanently_park_a_continuation(self):
        scheduler, _, running, adder = self.parked_continuation_scheduler()
        scheduler.chunked_prefill_size = 8192
        scheduler.truncation_align_size = 4096
        scheduler.min_chunked_prefill_size = 4096
        scheduler.dynamic_chunk_sizer = MagicMock()
        scheduler.dynamic_chunk_sizer.predict.return_value = 2048
        with (
            get_context().override_server_args(),
            patch(
                "sglang.srt.managers.scheduler.PrefillAdder", return_value=adder
            ) as factory,
            patch("sglang.srt.managers.scheduler.ScheduleBatch.init_new"),
            patch("sglang.srt.managers.scheduler.PrefillStats.from_adder"),
        ):
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running)
        self.assertEqual(factory.call_args.args[6], 4096)


if __name__ == "__main__":
    unittest.main()
