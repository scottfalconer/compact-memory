from CompressionStrategy.contrib.prototype_system_utils import (
    render_five_w_template,
)


def test_render_five_w_template():
    out = render_five_w_template("hello", who="alice", why="greet")
    assert out.startswith("WHO: alice;")
    assert "WHY: greet." in out
    assert "WHAT:" not in out
    assert out.endswith("CONTENT: hello")
