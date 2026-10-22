import subprocess
from pathlib import Path

import pytest

from scripts.checkpoint import git, rollback, sync


def test_update_check_rollback_and_dirty_guard(tmp_path):
    origin = tmp_path/'origin'
    origin.mkdir()
    subprocess.run(['git','init','-b','main',str(origin)],check=True,capture_output=True)
    git(origin,'config','user.name','Test')
    git(origin,'config','user.email','test@example.invalid')
    (origin/'file.txt').write_text('v1')
    git(origin,'add','file.txt');git(origin,'commit','-m','First version')
    first=git(origin,'rev-parse','HEAD')
    checkout=tmp_path/'checkout'
    subprocess.run(['git','clone',str(origin),str(checkout)],check=True,capture_output=True)
    (origin/'file.txt').write_text('v2')
    git(origin,'add','file.txt');git(origin,'commit','-m','Second version')
    second=git(origin,'rev-parse','HEAD')
    data=tmp_path/'drive'
    sync(checkout,data,'main',check=True)
    assert git(checkout,'rev-parse','HEAD')==first
    sync(checkout,data,'main')
    assert git(checkout,'rev-parse','HEAD')==second
    assert (data/'checkpoints'/f'{first}.bundle').is_file()
    rollback(checkout,data,first)
    assert (checkout/'file.txt').read_text()=='v1'
    (checkout/'file.txt').write_text('local edit')
    with pytest.raises(RuntimeError,match='Yerel'):
        sync(checkout,data,'main')
    assert (checkout/'file.txt').read_text()=='local edit'
