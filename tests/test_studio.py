import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ozzy.app import create_app
from ozzy.engine import Engine
from ozzy.schema import Generation
from ozzy.storage import Store, composite_mask

TOKEN = "test-access-key-with-at-least-32-characters"


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, Engine(demo=True), TOKEN)
    with TestClient(app) as c:
        yield c


def png(color="red", size=(256, 256)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def auth(client):
    client.headers['Authorization'] = 'Bearer ' + TOKEN


def wait_job(client, ident):
    for _ in range(100):
        job = client.get('/api/jobs/' + ident).json()
        if job['status'] not in {'queued', 'running'}:
            return job
        time.sleep(.03)
    pytest.fail('Job did not finish')


def test_auth_and_csrf(client):
    assert client.get('/api/jobs').status_code == 401
    assert client.post('/mcp/').status_code == 401
    assert client.post('/auth/login', json={'token': 'wrong'}).status_code == 401
    assert client.post('/auth/login', json={'token': TOKEN}).status_code == 200
    assert client.get('/api/jobs').status_code == 200
    assert client.post('/api/jobs', json={'prompt': 'x'}).status_code == 403
    assert client.post('/api/jobs', json={'prompt': 'x'}, headers={'Origin': 'https://evil.example', 'X-Ozzy-Client':'studio'}).status_code == 403


def test_generation_persistence_and_recipe(client):
    auth(client)
    response = client.post('/api/jobs', json={'prompt':'A cup', 'seed':42, 'width':256, 'height':256, 'count':2})
    assert response.status_code == 202
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'done'
    assert [o['seed'] for o in job['outputs']] == [42,43]
    item = job['outputs'][0]
    assert client.get(item['url']).headers['content-type'] == 'image/png'
    assert client.get(item['url']+'/recipe').json()['request']['prompt'] == 'A cup'
    assert client.post(item['url']+'/reuse').status_code == 200
    assert client.app.state.studio.store.read('jobs', job['id'])['status'] == 'done'


def test_validation_and_mask(client):
    auth(client)
    assert client.post('/api/jobs', json={'prompt':'x','width':257}).status_code == 422
    assert client.post('/api/jobs', json={'prompt':'x','mode':'edit'}).status_code == 422
    asset = client.post('/api/assets', content=png()).json()['id']
    selection = client.post('/api/assets', content=png('black')).json()['id']
    assert client.post('/api/jobs', json={'prompt':'x','mode':'mask','references':[asset],'mask':selection}).status_code == 400
    assert client.post('/api/jobs', json={'prompt':'x','mode':'edit','references':['../secret']}).status_code == 400


def test_mask_pixel_preservation(tmp_path):
    base, output, mask = [tmp_path/n for n in ('base.png','out.png','mask.png')]
    Image.new('RGB',(32,32),'red').save(base)
    Image.new('RGB',(32,32),'blue').save(output)
    selection = Image.new('L',(32,32),0)
    selection.putpixel((16,16),255)
    selection.save(mask)
    composite_mask(base,output,mask,0)
    with Image.open(output) as image:
        assert image.getpixel((0,0)) == (255,0,0,255)
        assert image.getpixel((16,16)) == (0,0,255,255)


def test_cancel(client):
    auth(client)
    job = client.post('/api/jobs', json={'prompt':'x','width':256,'height':256,'count':8}).json()
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    assert wait_job(client,job['id'])['status'] == 'cancelled'


def test_presets(client):
    auth(client)
    item=client.post('/api/presets',json={'name':'Portrait','settings':{'prompt':'A portrait'}}).json()
    assert client.get('/api/presets').json()[0]['id'] == item['id']
    assert client.delete('/api/presets/'+item['id']).status_code == 200
    assert client.get('/api/presets').json() == []


def test_mcp_transport_and_tools(client):
    auth(client)
    headers={'Accept':'application/json, text/event-stream'}
    response=client.post('/mcp/',headers=headers,json={'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-03-26','capabilities':{},'clientInfo':{'name':'test','version':'1'}}})
    assert response.status_code == 200
    response=client.post('/mcp/',headers=headers,json={'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})
    names={tool['name'] for tool in response.json()['result']['tools']}
    assert {'generate_image','upload_image','get_image','cancel_job','get_recipe','get_workflow','save_preset'} <= names
    response=client.post('/mcp/',headers=headers,json={'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'generate_image','arguments':{'settings':{'prompt':'MCP cup','width':256,'height':256}}}})
    result=response.json()['result']
    assert not result.get('isError',False)


def test_workflow_keeps_prompt_as_data():
    from ozzy.engine import workflow
    request=Generation(prompt='$(touch hacked); --help')
    graph=workflow(request,42,[],'ozzy/test')
    assert graph['4']['inputs']['prompt']==request.prompt
    assert graph['6']['inputs']['device']=='gpu'
    assert graph['7']['inputs']['cfg']==1


def test_notebook_is_clean_and_valid():
    import nbformat
    notebook=nbformat.read('notebooks/OzzyImage_Lab.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.outputs == []
            compile(cell.source,'notebook','exec')


def test_restart_marks_incomplete_jobs_interrupted(tmp_path):
    from ozzy.storage import atomic_json
    from ozzy.service import Studio
    store=Store(tmp_path)
    ident='a'*32
    atomic_json(store.path('jobs',ident,'.json'),{'id':ident,'status':'running','outputs':[]})
    studio=Studio(store,Engine(demo=True))
    try:
        assert studio.get(ident)['status']=='interrupted'
        assert store.read('jobs',ident)['status']=='interrupted'
    finally:
        studio.close()


def test_invalid_upload_is_rejected(client):
    auth(client)
    assert client.post('/api/assets',content=b'not an image').status_code==400


def test_workflow_export_and_demo_absence(client):
    auth(client)
    from ozzy.storage import atomic_json
    store=client.app.state.studio.store
    output_id, job_id='b'*32,'c'*32
    atomic_json(store.path('outputs',output_id,'.json'),{'job_id':job_id})
    assert client.get(f'/api/outputs/{output_id}/workflow').status_code==400
    folder=store.root/'jobs'/job_id
    folder.mkdir()
    atomic_json(folder/f'workflow-{output_id}.json',{'1':{'class_type':'UNETLoader','inputs':{}}})
    assert client.get(f'/api/outputs/{output_id}/workflow').json()['1']['class_type']=='UNETLoader'
    client.headers.pop('Authorization')
    assert client.get(f'/api/outputs/{output_id}/workflow').status_code==401
