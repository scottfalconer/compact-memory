from gist_memory.embedder import RandomEmbedder


def test_random_embedder_dim_and_seed():
    embedder = RandomEmbedder(dim=32, seed=42)
    vec = embedder.embed("test")
    assert len(vec) == 32

    embedder2 = RandomEmbedder(dim=32, seed=42)
    vec2 = embedder2.embed("test")
    assert (vec == vec2).all()
