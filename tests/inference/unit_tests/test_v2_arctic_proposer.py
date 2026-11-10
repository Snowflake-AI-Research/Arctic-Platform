"""Arctic proposal selection against v2 request slots and verification rows."""

from types import SimpleNamespace

import numpy as np
import torch

from arctic_platform.inference.suffix_decoding import SuffixDecodingCache
from arctic_platform.inference.vllm.spec_dec.arctic_proposer import ArcticProposer, SuffixProposer


def _config(**overrides):
    values = dict(num_speculative_tokens=4, disable_by_batch_size=2,
                  suffix_cache_max_depth=16, suffix_cache_max_requests=100,
                  suffix_max_spec_factor=1.0, suffix_max_spec_offset=0.0,
                  suffix_min_token_prob=0.1)
    values.update(overrides)
    return SimpleNamespace(**values)


class _Draft:
    n_predict = 3

    def generate_proposals(self, input_ids, previous_hidden_states, num_predict_tokens):
        self.context_tokens = input_ids.clone()
        self.hidden_states = previous_hidden_states.clone()
        self.calls = getattr(self, 'calls', 0) + 1
        return input_ids[:, None].expand(-1, num_predict_tokens)


def _neural():
    proposer = ArcticProposer.__new__(ArcticProposer)
    proposer.device = torch.device('cpu')
    proposer.speculative_config = _config()
    proposer.num_speculative_steps = 4
    proposer.model = _Draft()
    proposer.suffix = None
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=128))
    proposer.req_states = SimpleNamespace(total_len=SimpleNamespace(gpu=torch.tensor([23, 40, 10])))
    return proposer


def _propose(proposer, batch, hidden, sampled, rejected, last_sampled, **kwargs):
    return proposer.propose(batch, {}, {}, hidden, None, sampled, rejected,
                            last_sampled, None, None, None, **kwargs)


def test_neural_uses_last_accepted_hidden_row_and_request_slot_mapping():
    proposer = _neural()
    batch = SimpleNamespace(num_reqs=3, req_ids=['c', 'a', 'b'],
                            query_start_loc=torch.tensor([0, 3, 7, 10]),
                            idx_mapping=torch.tensor([2, 0, 1]))
    hidden = torch.arange(20).reshape(10, 2)
    drafts = _propose(proposer, batch, hidden, torch.tensor([2, 1, 3]),
                      torch.tensor([1, 3, 0]), torch.tensor([[101], [102], [103]]))
    assert torch.equal(proposer.model.hidden_states, hidden[[1, 3]])
    assert proposer.model.context_tokens.tolist() == [103, 101]
    assert drafts.tolist() == [[103, 103, 103, -1], [101, 101, 101, -1], [-1] * 4]
    assert proposer.last_draft_lengths == {'c': 3, 'a': 3, 'b': 0}


def test_neural_masks_partial_prefill_and_limits_model_length():
    proposer = _neural()
    proposer.req_states.total_len.gpu[0] = 126
    batch = SimpleNamespace(num_reqs=2, req_ids=['a', 'b'],
                            query_start_loc=torch.tensor([0, 2, 4]),
                            idx_mapping=torch.tensor([0, 1]))
    drafts = _propose(proposer, batch, torch.arange(8).reshape(4, 2),
                      torch.tensor([1, 0]), torch.tensor([1, 0]),
                      torch.tensor([[101], [102]]))
    assert drafts.tolist() == [[101, -1, -1, -1], [-1] * 4]


def test_neural_skip_drafting_does_not_execute_the_model():
    proposer = _neural()
    batch = SimpleNamespace(num_reqs=1, req_ids=['a'], idx_mapping=torch.tensor([0]))
    drafts = _propose(proposer, batch, None, torch.tensor([1]), None,
                      None, num_speculative_tokens=0)
    assert drafts.tolist() == [[-1] * 4]
    assert not hasattr(proposer.model, 'calls')


def test_suffix_preserves_cache_algorithm_and_does_not_duplicate_resumed_history():
    config = _config(num_speculative_tokens=3)
    prompt = [1, 2, 3, 1, 2, 3]
    response = [1, 2, 3]
    states = SimpleNamespace(
        req_id_to_index={'a': 0},
        total_len=SimpleNamespace(gpu=torch.tensor([len(prompt + response)])),
        prompt_len=SimpleNamespace(np=np.array([len(prompt)])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([prompt + response + [0] * 23])),
    )
    proposer = SuffixProposer(SimpleNamespace(speculative_config=config,
                                              model_config=SimpleNamespace(max_model_len=32)),
                              torch.device('cpu'), states)
    batch = SimpleNamespace(num_reqs=1, req_ids=['a'], idx_mapping=torch.tensor([0]))
    reference = SuffixDecodingCache(max_tree_depth=16, max_cached_requests=100)
    reference.start_request('a', prompt)
    reference.add_active_response('a', response)
    expected = reference.speculate('a', prompt + response, max_spec_tokens=3,
                                   max_spec_factor=1, max_spec_offset=0, min_token_prob=0.1).token_ids
    first = _propose(proposer, batch, None, torch.tensor([1]), None, None)
    assert first.tolist() == [expected + [-1] * (3 - len(expected))]
    resumed = _propose(proposer, batch, None, torch.tensor([1]), None, None)
    assert torch.equal(resumed, first)
    proposer.on_requests_finished({'a'})
    assert 'a' not in proposer.cache.active_requests
    assert 'a' in proposer.cache.cached_requests
    assert not proposer._lengths
    proposer.reset_state()
    assert not proposer.cache.cached_requests
