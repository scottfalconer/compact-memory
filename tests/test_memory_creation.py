from gist_memory.memory_creation import IdentityMemoryCreator


def test_identity_memory_creator():
    creator = IdentityMemoryCreator()
    text = "hello world"
    assert creator.create(text) == text
