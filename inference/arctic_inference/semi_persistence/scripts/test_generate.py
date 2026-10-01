from arctic_inference.semi_persistence import Orchestrator as orch


def wait_all() -> None:
    for mid in orch.models():
        orch.wait(mid)


def main() -> None:
    from time import sleep

    orch.init("/data-fast/image-cache/demo", [4, 5, 6])

    orch.register("model 1", {"model": "Qwen/Qwen3.5-35B-A3B",           "gpu_memory_utilization": 0.8})
    orch.register("model 2", {"model": "Qwen/Qwen3.5-35B-A3B-FP8",       "gpu_memory_utilization": 0.8})
    orch.register("model 3", {"model": "Qwen/Qwen3.5-35B-A3B-Base",      "gpu_memory_utilization": 0.8})
    orch.register("model 4",  {"model": "Qwen/Qwen3.5-27B",              "gpu_memory_utilization": 0.8})
    orch.register("model 5",  {"model": "Qwen/Qwen3.5-27B-FP8",          "gpu_memory_utilization": 0.8})
    orch.register("model 6",  {"model": "Qwen/Qwen3.5-9B",               "gpu_memory_utilization": 0.8})
    orch.register("model 7",  {"model": "Qwen/Qwen3.5-9B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 8",  {"model": "Qwen/Qwen3.5-4B",               "gpu_memory_utilization": 0.8})
    orch.register("model 9",  {"model": "Qwen/Qwen3.5-4B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 10", {"model": "Qwen/Qwen3.5-2B",               "gpu_memory_utilization": 0.8})
    orch.register("model 11", {"model": "Qwen/Qwen3.5-2B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 12", {"model": "Qwen/Qwen3.5-0.8B",             "gpu_memory_utilization": 0.8})
    orch.register("model 13", {"model": "Qwen/Qwen3.5-0.8B-Base",        "gpu_memory_utilization": 0.8})
    orch.register("model 14", {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","gpu_memory_utilization": 0.8})
    orch.register("model 15", {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4",    "gpu_memory_utilization": 0.8})

    for i in range(2):
        for j in range(3):
            orch.generate("model 2", "who is Elvis Presley?", 1000)
            orch.generate("model 4", "who is Celine Dion?", 1000)
            orch.generate("model 5", "who is Murpyh?", 1000)
            orch.generate("model 6", "who is the presitent of the United States?", 1000)
            orch.generate("model 7", "what is your name?", 1000)
            orch.generate("model 8", "where is greenland?", 1000)
            orch.generate("model 9", "give me a recipe", 1000)
            orch.generate("model 10", "tell me a joke", 1000)
            orch.generate("model 11", "who am I?", 1000)
            orch.generate("model 12", "who are you?", 1000)
            orch.generate("model 13", "what is the capital of france?", 1000)
            orch.generate("model 14", "what is the capital of germany?", 1000)
            orch.generate("model 15", "what is the capital of italy?", 1000)

        wait_all()
        sleep(15)

    orch.move("model 2", "saved")
    orch.move("model 4", "saved")
    orch.move("model 5", "saved")
    orch.move("model 6", "saved")
    orch.move("model 7", "saved")
    orch.move("model 8", "saved")
    orch.move("model 9", "saved")
    orch.move("model 10", "saved")
    orch.move("model 11", "saved")
    orch.move("model 12", "saved")
    orch.move("model 13", "saved")
    orch.move("model 14", "saved")
    orch.move("model 15", "saved")

    wait_all()

if __name__ == "__main__":
    main()
