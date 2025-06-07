from compact_memory.validation.llm_judge_metric import LLMJudgeMetric
import openai


class DummyClient:
    def __init__(self):
        self.call_count = 0
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature=0):
        self.call_count += 1

        class R:
            choices = [
                type("Msg", (), {"message": type("M", (), {"content": "0.8"})()})
            ]

        return R()


def test_llm_judge_metric(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: dummy)
    metric = LLMJudgeMetric(model_name="gpt-x")
    res1 = metric.evaluate(llm_response="foo", reference_answer="foo")
    res2 = metric.evaluate(llm_response="foo", reference_answer="foo")
    assert res1["llm_judge_score"] == 0.8
    assert dummy.call_count == 1
    assert res2["llm_judge_score"] == 0.8
