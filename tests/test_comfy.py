import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from PIL import Image
from pydantic import ValidationError
from ozzy.comfy_runtime import GPU_FLAGS, require_gpu, validate_nodes
from ozzy.engine import Engine, MODEL_FILES, workflow
from ozzy.schema import Generation
from ozzy.storage import Store


def test_gpu_only_contract():
    assert {'--gpu-only','--disable-async-offload','--disable-dynamic-vram','--bf16-unet','--bf16-text-enc','--bf16-vae'} <= set(GPU_FLAGS)
    assert not Generation(prompt='x').offload
    with pytest.raises(ValidationError):
        Generation(prompt='x',offload=True)


@pytest.mark.parametrize('gib,accepted',[(40,False),(80,True)])
def test_gpu_capacity_gate(monkeypatch,gib,accepted):
    import sys
    fake=SimpleNamespace(__version__='test',cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,get_device_properties=lambda _:SimpleNamespace(name='A100',total_memory=gib*1024**3)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    if accepted: assert require_gpu()['memory_gib']==80
    else:
        with pytest.raises(RuntimeError,match='80 GB'): require_gpu()


def test_reference_workflow_and_transparency():
    req=Generation(prompt='extract the cup',mode='extract',references=['a'*32],vae_tiling=True)
    graph=workflow(req,42,['one.png','two.png'],'ozzy/test')
    assert graph['4']['inputs']['images.image_2']==['202',0]
    assert graph['201']['class_type']=='JoinImageWithAlpha'
    assert graph['201']['inputs']['alpha']==['101',1]
    assert graph['7']['inputs']['latent_image']==['4',2]
    assert graph['8']['class_type']=='VAEDecodeTiled'
    assert 'alpha channel' in graph['4']['inputs']['prompt']
    assert graph['1']['inputs']['unet_name']==MODEL_FILES['denoiser']
    assert graph['2']['inputs']['clip_name']==MODEL_FILES['encoder']


def test_reference_preparation_preserves_alpha(tmp_path):
    store=Store(tmp_path/'data'); buffer=io.BytesIO()
    Image.new('RGBA',(100,200),(255,0,0,64)).save(buffer,format='PNG')
    asset=store.upload(buffer.getvalue())
    paths=Engine(demo=True).prepare_references(Generation(prompt='x',mode='edit',references=[asset['id']],width=256,height=512),store,tmp_path)
    with Image.open(paths[0]) as image:
        assert image.size==(256,512)
        assert image.getchannel('A').getextrema()==(64,64)


@pytest.mark.parametrize('outcome',['success','cancel','oom','reject'])
def test_comfy_http_lifecycle(tmp_path,outcome):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'backend':'comfyui','gpu_only':True,'comfy_url':'http://127.0.0.1:8188'}))
    cancel=threading.Event(); seen=[]; active={}; buffer=io.BytesIO()
    Image.new('RGBA',(256,256),(255,0,0,64)).save(buffer,format='PNG')
    def respond(request):
        path=request.url.path
        body=json.loads(request.content) if request.method=='POST' else {}
        seen.append((path,body))
        if path=='/system_stats':return httpx.Response(200,json={'devices':[]})
        if path=='/prompt':
            active['id']=body['prompt_id']
            if outcome=='reject':return httpx.Response(400,json={'error':'bad workflow'})
            if outcome=='cancel':cancel.set()
            return httpx.Response(200,json={'prompt_id':active['id']})
        if path.startswith('/history/'):
            entry={'status':{'completed':True,'status_str':'success'},'outputs':{'9':{'images':[{'filename':'output.png','subfolder':'ozzy','type':'output'}]}}}
            if outcome=='oom':entry={'status':{'status_str':'error','messages':[['execution_error',{'exception_message':'CUDA out of memory'}]]}}
            return httpx.Response(200,json={active['id']:entry})
        if path=='/view':return httpx.Response(200,content=buffer.getvalue())
        if path in {'/queue','/interrupt'}:return httpx.Response(200,json={})
        raise AssertionError(path)
    engine=Engine(config,transport=httpx.MockTransport(respond))
    assert engine.capabilities()['config_ready']
    output=tmp_path/'out.png'
    args=(Generation(prompt='x',width=256,height=256),Store(tmp_path/'data'),output,42,tmp_path,cancel,lambda _:None)
    if outcome=='success':
        engine.run(*args)
        assert output.read_bytes()==buffer.getvalue()
        assert (tmp_path/'workflow-out.json').exists()
        assert not any(path=='/interrupt' for path,_ in seen)
    else:
        with pytest.raises(InterruptedError if outcome=='cancel' else RuntimeError):engine.run(*args)
        assert ('/interrupt',{'prompt_id':active['id']}) in seen
        assert ('/queue',{'delete':[active['id']]}) in seen
        assert not output.exists()


def test_missing_node_fails_startup():
    with pytest.raises(RuntimeError,match='eksik'):validate_nodes({})


def test_notebook_gpu_only_migration():
    nb=json.loads(Path('notebooks/OzzyImage_Lab.ipynb').read_text(encoding='utf-8'))
    source='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'start_comfy' in source and 'stop_process(comfy_process, comfy_log)' in source
    assert 'ENGINE_MODE' not in source and 'PROFILE =' not in source and 'ENCODER =' not in source
    assert 'setup_colab.py' in source and 'model-lock-bf16.json' in source
