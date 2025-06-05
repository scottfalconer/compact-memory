from pathlib import Path
from itertools import product
import numpy as np
import yaml
import tiktoken
from compact_memory.history_experiment import (
    HistoryExperimentConfig,
    run_history_experiment,
)
from compact_memory.response_experiment import (
    ResponseExperimentConfig,
    run_response_experiment,
)
from compact_memory.embedding_pipeline import MockEncoder
import compact_memory.response_experiment as rexp
import compact_memory.local_llm as local_llm
import compact_memory.embedding_pipeline as emb


class DummyTok:
    def encode(self, text):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


tiktoken.get_encoding = lambda name="gpt2": DummyTok()

# ---------------------------------------------------------------------------
# Patch response_experiment to run offline: use dummy LLM and mock embeddings.


class DummyLLM:
    def __init__(self, *a, **k):
        pass

    tokenizer = staticmethod(
        lambda text, return_tensors=None, truncation=None, max_length=None: {
            "input_ids": text.split()
        }
    )
    model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
    max_new_tokens = 10

    def prepare_prompt(self, agent, prompt, **kw):
        return prompt

    def reply(self, prompt: str) -> str:
        if "module" in prompt:
            return "B"
        if "secret" in prompt:
            return "123"
        return ""


# Patch modules
rexp.LocalChatModel = DummyLLM
local_llm.LocalChatModel = DummyLLM
emb._MODEL = None
emb._load_model = lambda *a, **k: MockEncoder()
emb.embed_text = lambda text, **kw: (
    MockEncoder().encode(text)
    if isinstance(text, str)
    else np.stack([MockEncoder().encode(t) for t in text])
)
rexp.MockEncoder = MockEncoder

# ---------------------------------------------------------------------------


def main() -> None:
    dataset_hist = Path("tests/data/history_dialogues.yaml")
    dataset_resp = Path("tests/data/response_dialogues.yaml")

    param_grid = []
    for fr, mo, thr, boost in product([1, 2], [1, 2], [0.0, 0.1], [1.0, 1.5]):
        param_grid.append(
            {
                "config_prompt_num_forced_recent_turns": fr,
                "config_prompt_max_activated_older_turns": mo,
                "config_prompt_activation_threshold_for_inclusion": thr,
                "config_relevance_boost_factor": boost,
            }
        )

    hist_cfg = HistoryExperimentConfig(dataset=dataset_hist, param_grid=param_grid)
    hist_results = run_history_experiment(hist_cfg)

    resp_cfg = ResponseExperimentConfig(dataset=dataset_resp, param_grid=param_grid)
    resp_results = run_response_experiment(resp_cfg)

    output = {
        "history_results": hist_results,
        "response_results": resp_results,
    }
    print(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
