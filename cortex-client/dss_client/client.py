from .engine import ArcticInferenceEngine, DSSTrainingEngine


class DSSClient:
    def __init__(self, dss_server_url):
        self.dss_server_url = dss_server_url


class DSSTrainingClient(DSSClient):
    def __init__(self, dss_server_url):
        super().__init__(dss_server_url)

    def initialize(self, model, ds_config, training_config, lr_scheduler=None):
        return DSSTrainingEngine(
            model=model,
            ds_config=ds_config,
            training_config=training_config,
            dss_server_url=self.dss_server_url,
            lr_scheduler=lr_scheduler
        )


class ArcticInferenceClient(DSSClient):
    def __init__(self, dss_server_url):
        super().__init__(dss_server_url)

    def initialize(
        self,
        model_name: str,
        vllm_config: dict | None = None,
    ) -> ArcticInferenceEngine:
        return ArcticInferenceEngine(
            server_url=self.dss_server_url,
            model_name=model_name,
            vllm_config=vllm_config,
        )
