const $ = id => document.getElementById(id);
let mode = 'generate', references = [], activeJob = null, presets = [], erasing = false, drawing = false;
let maskHistory = [], lastPoint = null, busy = false;
const mask = $('maskCanvas'), ctx = mask.getContext('2d');
const maskData = document.createElement('canvas'), maskCtx = maskData.getContext('2d');
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; setTimeout(() => $('toast').hidden = true, 6000); }
async function api(url, options = {}) {
  const res = await fetch(url, {...options, headers: {'X-Ozzy-Client':'studio', ...options.headers}});
  if (res.status === 401) { if (!$('loginDialog').open) $('loginDialog').showModal(); throw new Error('Lütfen oturum açın.'); }
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail || res.status)); }
  return res.json();
}
const post = (url, data) => api(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
function setMode(value) {
  mode = value; document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('selected', b.dataset.mode === mode));
  $('referenceSection').hidden = !['edit','mask','extract'].includes(mode);
  $('maskToolbar').hidden = $('maskNote').hidden = mode !== 'mask';
  $('stageTitle').textContent = mode === 'mask' ? 'Maske tuvali' : 'Tuval';
  $('generate').firstChild.textContent = ({generate:'Görsel oluştur ',edit:'Referansla oluştur ',mask:'Seçili alanı düzenle ',transparent:'Şeffaf görsel oluştur ',extract:'Konuyu ayıkla '})[mode];
  if (mode === 'mask' && references.length) loadCanvas().catch(e => toast(e.message));
  else $('maskEditor').hidden = true;
}
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.querySelectorAll('[data-prompt]').forEach(b => b.onclick = () => $('prompt').value = b.dataset.prompt);
$('size').onchange = () => $('customSize').hidden = $('size').value !== 'custom';
function settings() {
  const dimensions = $('size').value === 'custom' ? [+$('width').value, +$('height').value] : $('size').value.split(',').map(Number);
  return {mode,prompt:$('prompt').value,negative_prompt:$('negative').value,width:dimensions[0],height:dimensions[1],steps:+$('steps').value,cfg:+$('cfg').value,seed:+$('seed').value,count:+$('count').value,references:['edit','mask','extract'].includes(mode)?references.map(r=>r.id):[],offload:false,vae_tiling:$('tiling').checked,prefix_cache:$('prefix').checked,preserve_unmasked:$('preserve').checked,mask_blur:+$('blur').value};
}
async function upload(file) { return api('/api/assets',{method:'POST',body:file}); }
function renderReferences() {
  $('references').replaceChildren();
  references.forEach((ref,index) => {
    const item=document.createElement('div'); item.className='ref';
    const img=document.createElement('img'); img.src=ref.url; img.alt=`Referans ${index+1}`; img.title=`Referans ${index+1}`;
    const remove=document.createElement('button'); remove.type='button';remove.textContent='×';remove.title='Referansı kaldır';
    remove.onclick=()=>{references.splice(index,1);renderReferences();if(mode==='mask') loadCanvas().catch(e=>toast(e.message));};
    item.append(img,remove);$('references').append(item);
  });
}
$('referenceFiles').onchange=async e=>{
  try { if(references.length+e.target.files.length>10)throw Error('En fazla 10 referans seçebilirsin.');
    for(const file of e.target.files)references.push(await upload(file));renderReferences();if(mode==='mask')await loadCanvas();
  }catch(e){toast(e.message);}finally{e.target.value='';}
};
async function loadImage(url){const img=new Image();img.src=url;await img.decode();return img;}
let canvasReference=null;
async function loadCanvas(){
  if(!references.length){$('maskEditor').hidden=true;$('emptyState').hidden=false;canvasReference=null;return;}
  if(canvasReference!==references[0].id){
    const ref=references[0],img=await loadImage(ref.url);if(references[0]?.id!==ref.id)return;
    for(const canvas of [$('baseCanvas'),mask,maskData]){canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;}
    $('baseCanvas').getContext('2d').drawImage(img,0,0);canvasReference=ref.id;maskHistory=[];clearMask();
  }
  $('emptyState').hidden=$('resultImage').hidden=true;$('maskEditor').hidden=false;
}
function redrawMask(){ctx.clearRect(0,0,mask.width,mask.height);ctx.drawImage(maskData,0,0);ctx.globalCompositeOperation='source-in';ctx.fillStyle='#ee5555';ctx.fillRect(0,0,mask.width,mask.height);ctx.globalCompositeOperation='source-over';}
function clearMask(){maskCtx.clearRect(0,0,maskData.width,maskData.height);redrawMask();}
function snapshotMask(){maskHistory.push(maskCtx.getImageData(0,0,maskData.width,maskData.height));if(maskHistory.length>6)maskHistory.shift();}
function point(e){const r=mask.getBoundingClientRect();return [(e.clientX-r.left)*mask.width/r.width,(e.clientY-r.top)*mask.height/r.height];}
function paint(e){const p=point(e),r=mask.getBoundingClientRect();maskCtx.globalCompositeOperation=erasing?'destination-out':'source-over';maskCtx.strokeStyle='white';maskCtx.fillStyle='white';maskCtx.lineWidth=+$('brush').value*mask.width/r.width;maskCtx.lineCap='round';maskCtx.lineJoin='round';maskCtx.beginPath();maskCtx.moveTo(...(lastPoint||p));maskCtx.lineTo(...p);maskCtx.stroke();maskCtx.beginPath();maskCtx.arc(...p,maskCtx.lineWidth/2,0,Math.PI*2);maskCtx.fill();lastPoint=p;redrawMask();}
mask.onpointerdown=e=>{snapshotMask();drawing=true;lastPoint=null;mask.setPointerCapture(e.pointerId);paint(e);};
mask.onpointermove=e=>{if(drawing)paint(e);};mask.onpointerup=mask.onpointercancel=()=>{drawing=false;lastPoint=null;};
$('eraser').onclick=()=>{erasing=!erasing;$('eraser').classList.toggle('selected',erasing);};
$('clearMask').onclick=()=>{snapshotMask();clearMask();};
$('undoMask').onclick=()=>{const prev=maskHistory.pop();if(prev){maskCtx.putImageData(prev,0,0);redrawMask();}};
$('maskFile').onchange=async e=>{let url;try{if(!references.length)throw Error('Önce kaynak görsel seçin.');await loadCanvas();url=URL.createObjectURL(e.target.files[0]);const img=await loadImage(url);if(img.width!==mask.width||img.height!==mask.height)throw Error('Maske kaynak görselle aynı boyutta olmalı.');snapshotMask();maskCtx.globalCompositeOperation='source-over';maskCtx.clearRect(0,0,mask.width,mask.height);maskCtx.drawImage(img,0,0);const data=maskCtx.getImageData(0,0,mask.width,mask.height);for(let i=0;i<data.data.length;i+=4){const a=(data.data[i]+data.data[i+1]+data.data[i+2])/3;data.data[i]=data.data[i+1]=data.data[i+2]=255;data.data[i+3]=a;}maskCtx.putImageData(data,0,0);redrawMask();}catch(e){toast(e.message);}finally{if(url)URL.revokeObjectURL(url);e.target.value='';}};
async function maskAsset(){if(!references.length||canvasReference!==references[0].id)throw Error('Maske tuvalini açıp bölge seçin.');const exportCanvas=document.createElement('canvas');exportCanvas.width=mask.width;exportCanvas.height=mask.height;const c=exportCanvas.getContext('2d');c.fillStyle='black';c.fillRect(0,0,mask.width,mask.height);c.drawImage(maskData,0,0);return upload(await new Promise(resolve=>exportCanvas.toBlob(resolve,'image/png')));}
$('generateForm').onsubmit=async e=>{e.preventDefault();if(busy)return;busy=true;$('generate').disabled=true;try{const data=settings();if(mode==='mask')data.mask=(await maskAsset()).id;const job=await post('/api/jobs',data);activeJob=job.id;showJob(job);toast('Üretim kuyruğuna eklendi.');}catch(e){toast(e.message);}finally{busy=false;$('generate').disabled=false;}};
function showResult(item){$('resultImage').src=item.url;$('resultImage').hidden=false;$('maskEditor').hidden=$('emptyState').hidden=true;$('download').href=item.url;$('download').hidden=false;$('stageMeta').textContent=`${item.width} × ${item.height} / SEED ${item.seed}`;}
function showJob(job){$('jobStatus').textContent=({queued:'Sırada',running:'Üretiliyor',done:'Tamamlandı',failed:'Üretim başarısız',cancelled:'İptal edildi',interrupted:'Oturum kesildi'})[job.status]+` · ${job.completed}/${job.request.count}`;$('log').textContent=job.message;$('cancelJob').hidden=!['queued','running'].includes(job.status);if(job.outputs.length&&mode!=='mask')showResult(job.outputs.at(-1));}
$('cancelJob').onclick=async()=>{try{if(activeJob)showJob(await post(`/api/jobs/${activeJob}/cancel`,{}));}catch(e){toast(e.message);}};
async function restore(data){
  $('prompt').value=data.prompt;$('negative').value=data.negative_prompt;$('steps').value=data.steps;$('cfg').value=data.cfg;$('seed').value=data.seed;$('count').value=data.count;
  const size=`${data.width},${data.height}`;$('size').value=[...$('size').options].some(o=>o.value===size)?size:'custom';$('width').value=data.width;$('height').value=data.height;$('size').onchange();
  $('tiling').checked=data.vae_tiling;$('prefix').checked=data.prefix_cache;$('preserve').checked=data.preserve_unmasked;$('blur').value=data.mask_blur;
  references=data.references.map(id=>({id,url:`/api/assets/${id}`}));renderReferences();setMode(data.mode);
  if(data.mode==='mask'&&data.mask){await loadCanvas();const img=await loadImage(`/api/assets/${data.mask}`);maskCtx.globalCompositeOperation='source-over';maskCtx.clearRect(0,0,mask.width,mask.height);maskCtx.drawImage(img,0,0);const pixels=maskCtx.getImageData(0,0,mask.width,mask.height);for(let i=0;i<pixels.data.length;i+=4){const v=pixels.data[i];pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=255;pixels.data[i+3]=v;}maskCtx.putImageData(pixels,0,0);redrawMask();}
}
function button(text,fn){const b=document.createElement('button');b.className='text-button';b.textContent=text;b.onclick=()=>Promise.resolve(fn()).catch(e=>toast(e.message));return b;}
async function refreshGallery(){const jobs=await api('/api/jobs');$('gallery').replaceChildren();let n=0;for(const job of jobs){for(const item of job.outputs){n++;const card=document.createElement('article');card.className='card';const img=document.createElement('img');img.src=item.url;img.alt=job.request.prompt;img.loading='lazy';img.onclick=()=>showResult(item);const title=document.createElement('p');title.textContent=job.request.prompt;const info=document.createElement('small');info.textContent=`${job.backend.demo?'DEMO · ':''}${item.width} × ${item.height} · ${item.has_transparency?'RGBA':'RGB'} · ${item.seed}`;const actions=document.createElement('div');actions.className='actions';actions.append(button('Tarifi yükle',()=>restore({...job.request,seed:item.seed})),button('Referans yap',async()=>{references=[await post(`/api/outputs/${item.id}/reuse`,{})];renderReferences();setMode('edit');window.scrollTo({top:0,behavior:'smooth'});}));card.append(img,title,info,actions);if(job.backend.backend==='comfyui'){const link=document.createElement('a');link.className='text-button';link.href=item.url+'/workflow';link.download='comfy-workflow.json';link.textContent='Workflow indir ↓';card.append(link);}$('gallery').append(card);}}
  if(!n){const p=document.createElement('p');p.className='help';p.textContent='İlk çalışman için tuval hazır.';$('gallery').append(p);}
  $('jobHistory').replaceChildren();
  for(const job of jobs){const row=document.createElement('div');row.className='job-row';const title=document.createElement('span');title.textContent=`${new Date(job.created*1000).toLocaleString('tr-TR')} · ${job.status} · ${job.completed}/${job.request.count} · ${job.request.prompt.slice(0,65)}`;row.append(title,button('İncele',()=>{activeJob=job.id;showJob(job);}),button('Tarifi yükle',()=>restore({...job.request,seed:job.seed})));if(['queued','running'].includes(job.status))row.append(button('İptal',async()=>{await post(`/api/jobs/${job.id}/cancel`,{});await refreshGallery();}));$('jobHistory').append(row);}
  const ongoing=jobs.find(j=>['queued','running'].includes(j.status));if(!activeJob&&ongoing)activeJob=ongoing.id;
}
async function loadPresets(){presets=await api('/api/presets');$('presets').replaceChildren(new Option('Kaydedilmiş tarifler',''));presets.forEach(p=>$('presets').add(new Option(p.name,p.id)));}
$('presets').onchange=()=>{const p=presets.find(p=>p.id===$('presets').value);if(p)restore(p.request).catch(e=>toast(e.message));};
$('savePreset').onclick=async()=>{try{const name=prompt('Tarif adı');if(!name)return;const data=settings();if(mode==='mask')data.mask=(await maskAsset()).id;await post('/api/presets',{name,settings:data});await loadPresets();toast('Tarif kaydedildi.');}catch(e){toast(e.message);}};
$('deletePreset').onclick=async()=>{try{if(!$('presets').value)return;await api(`/api/presets/${$('presets').value}`,{method:'DELETE'});await loadPresets();}catch(e){toast(e.message);}};
$('refreshGallery').onclick=()=>refreshGallery().catch(e=>toast(e.message));
$('loginDialog').addEventListener('cancel',e=>e.preventDefault());
$('loginForm').onsubmit=async e=>{e.preventDefault();try{await post('/auth/login',{token:$('accessToken').value});$('accessToken').value='';$('loginError').textContent='';$('loginDialog').close();await init();}catch(e){$('loginError').textContent=e.message;}};
$('logout').onclick=async()=>{await post('/auth/logout',{});location.reload();};
function openMcp(){$('mcpUrl').textContent=location.origin+'/mcp/';$('mcpDialog').showModal();}
$('mcpInfo').onclick=$('connectNav').onclick=openMcp;$('closeMcp').onclick=()=>$('mcpDialog').close();
$('galleryNav').onclick=()=>$('gallerySection').scrollIntoView({behavior:'smooth'});$('studioNav').onclick=()=>window.scrollTo({top:0,behavior:'smooth'});
async function init(){const c=await api('/api/capabilities');$('backendLabel').textContent=c.demo?'Demo stüdyo':c.config_ready?`BF16 · ComfyUI · 80 GB`:'Model kurulumu gerekli';$('demoBanner').hidden=!c.demo;await refreshGallery();await loadPresets();}
let polling=false;
setInterval(async()=>{if(polling||!activeJob||$('loginDialog').open)return;polling=true;try{const job=await api(`/api/jobs/${activeJob}`);showJob(job);if(!['queued','running'].includes(job.status)){activeJob=null;if(job.outputs.length)showResult(job.outputs.at(-1));await refreshGallery();}}catch(e){toast(e.message);}finally{polling=false;}},2000);
init().catch(e=>{if(!$('loginDialog').open)toast(e.message);});
