from scripts import engine_cache as cache


def test_cache_round_trip_and_corruption(tmp_path, monkeypatch):
    source = tmp_path / 'build' / 'bin'
    source.mkdir(parents=True)
    binary = source / 'sd-cli'
    binary.write_bytes(b'test executable')
    (source / 'libtest.so').write_bytes(b'test library')
    environment = {'commit': 'example', 'cuda': '12', 'architecture': '80'}
    root = tmp_path / 'drive'
    messages = []
    validated = []
    monkeypatch.setattr(cache, 'validate', lambda path: validated.append(path))
    assert cache.restore(root, tmp_path, environment, messages.append) is None
    cache.save(root, binary, environment, messages.append)
    restored = cache.restore(root, tmp_path, environment, messages.append)
    assert restored.read_bytes() == binary.read_bytes()
    assert (restored.parent / 'libtest.so').read_bytes() == b'test library'
    assert validated == [restored]
    assert cache.restore(root, tmp_path, {**environment, 'cuda': '13'}, messages.append) is None
    (root / cache.cache_key(environment) / 'bin' / 'sd-cli').write_bytes(b'corrupt')
    assert cache.restore(root, tmp_path, environment, messages.append) is None


def test_cache_rejects_incompatible_executable(tmp_path, monkeypatch):
    binary = tmp_path / 'build' / 'sd-cli'
    binary.parent.mkdir()
    binary.write_bytes(b'test')
    env = {'commit': 'example'}
    cache.save(tmp_path / 'cache', binary, env, lambda _: None)
    def fail(path):
        raise RuntimeError('missing CUDA device')
    monkeypatch.setattr(cache, 'validate', fail)
    assert cache.restore(tmp_path / 'cache', tmp_path, env, lambda _: None) is None
