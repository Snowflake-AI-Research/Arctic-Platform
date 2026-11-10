from types import SimpleNamespace

from arctic_platform.inference.semi_persistence.ca_graph_rebind import _graph_manager_holders


def test_all_v2_graph_holders_and_lazy_arctic_proposals_are_discovered():
    base = SimpleNamespace(graphs={})
    shift = SimpleNamespace(graphs={})
    native_draft = SimpleNamespace(graphs={})
    proposal_graphs = {}
    runner = SimpleNamespace(
        cudagraph_manager=base, shift_cudagraph_manager=shift,
        speculator=SimpleNamespace(cudagraph_manager=native_draft),
        get_draft_model=lambda: SimpleNamespace(cuda_graphs=proposal_graphs),
    )
    holders, why = _graph_manager_holders(SimpleNamespace(model_runner=runner))
    assert why is None
    assert holders[:3] == [base, shift, native_draft]
    proposal_graphs[4] = object()
    assert holders[3].graphs is proposal_graphs
    assert holders[3].graphs[4] is proposal_graphs[4]


def test_shared_graph_manager_is_not_enumerated_twice():
    manager = SimpleNamespace(graphs={})
    runner = SimpleNamespace(cudagraph_manager=manager, shift_cudagraph_manager=manager)
    assert _graph_manager_holders(SimpleNamespace(model_runner=runner)) == ([manager], None)
