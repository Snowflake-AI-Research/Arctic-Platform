"""Standalone CPU contract; import stubs require an isolated interpreter."""
import asyncio
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ('vllm', 'vllm.config', 'vllm.v1', 'vllm.v1.metrics',
             'vllm.v1.metrics.loggers', 'vllm.v1.metrics.stats', 'ray'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['vllm.config'].VllmConfig = object
sys.modules['vllm.v1.metrics.loggers'].StatLoggerBase = object
for name in ('SchedulerStats', 'IterationStats', 'MultiModalCacheStats'):
    setattr(sys.modules['vllm.v1.metrics.stats'], name, object)
sys.modules['ray'].remote = lambda cls: cls
package = types.ModuleType('arctic_platform.inference.server')
package.__path__ = [str(ROOT / 'arctic_platform/inference/server')]
sys.modules[package.__name__] = package
metrics = importlib.import_module(package.__name__ + '.metrics')
worker = importlib.import_module(package.__name__ + '.worker')
scheduler = importlib.import_module(package.__name__ + '.scheduler')
NS = types.SimpleNamespace


class SamplingCounterContract(unittest.TestCase):
    def test_iteration_totals_survive_thinning_eviction_and_repeated_drains(self):
        collector = metrics.WorkerMetricsCollector(max_snapshots=1, min_interval_s=1)
        metrics._COLLECTORS[0] = collector
        actor = worker.InferenceWorker.__new__(worker.InferenceWorker)

        async def drain():
            return actor.drain_metrics()

        handle = NS(drain_metrics=NS(remote=drain), set_replica_id=NS(remote=lambda _: None))
        sched = scheduler.Scheduler([handle], dynamic_concurrency=False)
        stats = NS(kv_cache_usage=.5, num_running_reqs=1, num_waiting_reqs=0,
                   spec_decoding_stats=NS(num_drafts=2, num_draft_tokens=6, num_accepted_tokens=3))
        iteration = NS(num_preempted_reqs=1, num_generation_tokens=4,
                       prompt_token_stats=NS(computed=0))
        with patch.object(metrics.time, 'time', side_effect=[100., 100.1, 102.]):
            for _ in range(3):
                collector.push_snapshot(stats, iteration)
        collector.push_snapshot(None, iteration)
        payload = asyncio.run(sched.drain_metrics())
        replica = payload['replicas'][0]
        self.assertEqual(len(replica['snapshots']), 1)
        self.assertEqual(replica['snapshots'][0]['max_concurrency'], 64)
        totals = replica['engine_totals']
        self.assertEqual({k: v for k, v in totals.items() if k != 'started_at'},
                         dict(num_preempted_reqs=4, num_drafts=6,
                              num_draft_tokens=18, num_accepted_tokens=9))
        again = asyncio.run(sched.drain_metrics())['replicas'][0]
        self.assertEqual(again['snapshots'], [])
        self.assertEqual(again['engine_totals'], totals)
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_request_metrics_are_serialized_only_when_vllm_supplies_them(self):
        detail = dict(num_draft_tokens=6, num_accepted_draft_tokens=3,
                      per_step_accepted=[1, 2], per_step_drafted=[3, 3])
        choice = NS(text='x', token_ids=[42], finish_reason='stop', logprobs=None,
                    spec_decode_metrics=NS(to_dict=lambda: detail))
        output = NS(outputs=[choice], prompt_token_ids=[41], num_cached_tokens=0,
                    prompt_logprobs=None)
        result = worker._result_from_output(output, return_sampled_logprobs_only=True)
        self.assertEqual(json.loads(json.dumps(result))['spec_decode_metrics'], detail)
        choice.spec_decode_metrics = None
        result = worker._result_from_output(output, return_sampled_logprobs_only=True)
        self.assertNotIn('spec_decode_metrics', result)

    def test_phase_timestamps_are_returned_on_the_wall_clock(self):
        choice = NS(text='x', token_ids=[42], finish_reason='stop', logprobs=None,
                    spec_decode_metrics=None)
        stats = NS(queued_ts=10., scheduled_ts=10.25, first_token_ts=11.5, last_token_ts=0.)
        output = NS(outputs=[choice], prompt_token_ids=[41], num_cached_tokens=0,
                    prompt_logprobs=None, metrics=stats)
        with patch.object(worker.time, 'time', return_value=1_000_000.), \
                patch.object(worker.time, 'monotonic', return_value=20.):
            result = worker._result_from_output(output, return_sampled_logprobs_only=True)
        # offset = 1_000_000 - 20; last_token_ts was never set, so it is left out.
        self.assertEqual({k: result.get(k) for k in worker._PHASE_TIMESTAMP_KEYS},
                         dict(queued_ts=999_990., scheduled_ts=999_990.25,
                              first_token_ts=999_991.5, last_token_ts=None))
        self.assertEqual(json.loads(json.dumps(result))['first_token_ts'], 999_991.5)
        output.metrics = None
        result = worker._result_from_output(output, return_sampled_logprobs_only=True)
        self.assertFalse(set(worker._PHASE_TIMESTAMP_KEYS) & set(result))

    def test_non_streaming_record_carries_the_first_token_time(self):
        replies = [dict(token_ids=[1], prompt_len=3, generation_len=1, prefix_cache_len=0,
                        first_token_ts=123.5),
                   dict(token_ids=[1], prompt_len=3, generation_len=1, prefix_cache_len=0)]

        async def generate(prompt, sampling_params):
            return replies.pop(0)

        handle = NS(generate=NS(remote=generate), set_replica_id=NS(remote=lambda _: None))
        sched = scheduler.Scheduler([handle], dynamic_concurrency=False)

        async def run():
            sched._ensure_background_tasks = lambda: None
            for _ in range(2):
                await sched.submit([1, 2, 3], {})
                await asyncio.sleep(0)
            return [r.first_delta_time for r in sched._request_records.drain()]

        self.assertEqual(asyncio.run(run()), [123.5, None])


if __name__ == '__main__':
    unittest.main()
