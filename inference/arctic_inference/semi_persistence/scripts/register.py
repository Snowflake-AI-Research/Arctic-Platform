from arctic_inference.semi_persistence import Orchestrator as orch

config_qwen3_8b = {
    "model": "Qwen/Qwen3-8B",
    "speculative_config": {
        "method": "arctic",
        "model": "/data-fast/spec-decode-qwen3-8b-search_r1",
    },
    "gpu_memory_utilization": 0.4,
    "_env": {"ARCTIC_INFERENCE_ENABLED": "1"},
}

config_qwen3_30b = {
    "model": "Qwen/Qwen3-30B-A3B",
    "speculative_config": {
        "method": "arctic",
        "model": "/data-fast/spec-decode-qwen3-30b-search_r1",
    },
    "gpu_memory_utilization": 0.8,
    "_env": {"ARCTIC_INFERENCE_ENABLED": "1"},
}

config_qwen32b_lcontext = {
    "model": "Qwen/Qwen3-32B",
    "speculative_config": {
        "method": "arctic",
        "model": "/data-fast/qwen3-32b-longcontext-4096-3head",
    },
    "gpu_memory_utilization": 0.8,
    "_env": {"ARCTIC_INFERENCE_ENABLED": "1"},
}

config_qwen32b_bird = {
    "model": "Qwen/Qwen3-32B",
    "speculative_config": {
        "method": "arctic",
        "model": "/data-fast/qwen3-32b-bird-4096-3head",
    },
    "gpu_memory_utilization": 0.8,
    "_env": {"ARCTIC_INFERENCE_ENABLED": "1"},
}

def main():

    orch.init("/data-fast/image-cache/test_image")

    # orch.register("model 1", {"model": "Qwen/Qwen3.5-35B-A3B",              "gpu_memory_utilization": 0.8})
    # orch.register("model 2", {"model": "Qwen/Qwen3.5-35B-A3B-FP8",          "gpu_memory_utilization": 0.8})
    # orch.register("model 3", {"model": "Qwen/Qwen3.5-35B-A3B-Base",         "gpu_memory_utilization": 0.8})
    # orch.register("model 4",  {"model": "Qwen/Qwen3.5-27B",                 "gpu_memory_utilization": 0.8})
    # orch.register("model 5",  {"model": "Qwen/Qwen3.5-27B-FP8",             "gpu_memory_utilization": 0.4, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 6",  {"model": "Qwen/Qwen3.5-9B",                  "gpu_memory_utilization": 0.4, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 7",  {"model": "Qwen/Qwen3.5-9B-Base",             "gpu_memory_utilization": 0.4, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    orch.register("model 8",  {"model": "Qwen/Qwen3.5-4B",                  "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}, "max_num_seqs": 512})
    # orch.register("model 9",  {"model": "Qwen/Qwen3.5-4B-Base",             "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    orch.register("model 10", {"model": "Qwen/Qwen3.5-2B",                  "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 11", {"model": "Qwen/Qwen3.5-2B-Base",             "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 12", {"model": "Qwen/Qwen3.5-0.8B",                "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 13", {"model": "Qwen/Qwen3.5-0.8B-Base",           "gpu_memory_utilization": 0.2, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 14", {"model": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", "gpu_memory_utilization": 0.8})
    # orch.register("model 15", {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",   "gpu_memory_utilization": 0.4, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})
    # orch.register("model 16", {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4",       "gpu_memory_utilization": 0.4, "_env": {"ARCTIC_INFERENCE_ENABLED": "1"}})

    orch.register("spec 8b", config_qwen3_8b)
    orch.register("spec 30b", config_qwen3_30b)
    orch.register("32b lctx", config_qwen32b_lcontext)
    orch.register("32b bird", config_qwen32b_bird)
    orch.wait()


if __name__ == "__main__":
    main()
