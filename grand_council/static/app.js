'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const uid = () => 'a' + Math.random().toString(36).slice(2,10);
const defaults = [
  {name:'Systems Architect', role:'Analyze system architecture, interfaces, scalability, integration cost, and alternatives.', temperature:0.35},
  {name:'Critical Skeptic', role:'Attack assumptions, expose hidden constraints, contradictions, weak claims, and failure scenarios.', temperature:0.55},
  {name:'Implementation Lead', role:'Assess practical implementation plans, complexity, delivery phases, technical debt, and developer experience.', temperature:0.4},
  {name:'Quality & Safety', role:'Evaluate edge cases, reliability, testing, security, resource budgets, and operational risks.', temperature:0.3},
];
const state = {
  page:'council', models:[], agents:defaults.map(a=>({...a,id:uid(),model_id:'demo'})),
  chair_id:null, inspected:null, jobs:[], sessions:[], run:null,
  lastMessageKey:'', lastReportKey:'', hardware:null, active:false
};
state.chair_id=state.agents[0].id;
let toastTimer;
function toast(message, error=false) {const node=$('toast'); node.textContent=message; node.className=(error?'show error':'show'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>node.className='',5500);}
async function api(path, method='GET', payload=undefined) {
  let resp;
  try {resp = await fetch('/api'+path, {method, headers:payload!==undefined?{'Content-Type':'application/json'}:{},body:payload!==undefined?JSON.stringify(payload):undefined});}
  catch (e){throw new Error('Cannot reach the local server. '+e.message);}
  let result; try{result=await resp.json();}catch(e){result={detail:'Unexpected server response'};}
  if(!resp.ok) {let d=result.detail||result.message||resp.statusText; if(Array.isArray(d)) d=d.map(x=>x.msg).join(', '); throw new Error(String(d));}
  return result;
}
function navigate(name){
  state.page=name;
  document.querySelectorAll('[data-nav]').forEach(x=>x.classList.toggle('active',x.dataset.nav===name));
  document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id==='page-'+name));
  $('crumb-name').textContent=({'council':'Council chamber','results':'Deliberations','models':'Model library','archives':'Archives'})[name];
  if(name==='archives') refreshArchives();
  if(name==='models') {refreshModels();refreshJobs();}
  if(name==='results') refreshRun();
  window.scrollTo({top:0,behavior:'instant'});
}
function modelOptions(current) {
  return state.models.map(m=>`<option value="${esc(m.id)}" ${m.id===current?'selected':''}>${esc(m.label)} · ${esc(m.backend)}</option>`).join('');
}
function renderAgents() {
  $('agents-list').innerHTML=state.agents.map((a,i)=>`<div class="agent-card" data-agent="${esc(a.id)}">
    <div class="agent-head"><div class="agent-avatar">${String(i+1).padStart(2,'0')}</div><div style="flex:1;min-width:0"><input class="agent-name" data-field="name" aria-label="Agent name" value="${esc(a.name)}" maxlength="80"/><input class="agent-role" data-field="role" aria-label="Agent mandate" value="${esc(a.role)}" maxlength="1200" title="Agent mandate" /></div><button class="agent-remove" data-remove="${esc(a.id)}" aria-label="Remove agent" title="Remove agent">×</button></div>
    <div class="agent-controls"><div><label>ASSIGNED MODEL</label><select class="select-field" data-field="model_id">${modelOptions(a.model_id)}</select></div><div><label>TEMPERATURE</label><input class="text-field" type="number" data-field="temperature" min="0" max="1.5" step=".05" value="${esc(a.temperature)}"/></div></div>
  </div>`).join('');
  $('chair-select').innerHTML=state.agents.map(a=>`<option value="${esc(a.id)}" ${a.id===state.chair_id?'selected':''}>${esc(a.name)}</option>`).join('');
  $('run-estimate').textContent=`${state.agents.length} agents · ${$('rounds').value} debate round${$('rounds').value==='1'?'':'s'} · ${state.agents.length * (1+2*Number($('rounds').value))+1} baseline calls (continuations may add calls)`;
}
async function refreshModels(){
  try{
    const models=await api('/models'); state.models=models;
    $('model-count').textContent=models.length-1;
    state.agents.forEach(a=>{if(!models.some(m=>m.id===a.model_id))a.model_id='demo';});
    renderAgents();
    $('models-list').innerHTML=models.map(m=>`<div class="installed-item"><div class="installed-title">${esc(m.label)}${m.loaded?'<span class="live-dot" title="Loaded in memory"></span>':''}</div><div class="installed-meta">${esc(m.backend.toUpperCase())}${m.repo?' · '+esc(m.repo):''}${m.revision?' · '+esc(m.revision.slice(0,10)):''}${m.filename?' · '+esc(m.filename):''}</div><div class="installed-actions"><button data-load="${esc(m.id)}">${m.loaded?'↻ Reload':'▶ Load'}</button>${m.id==='demo'?'':`<button class="danger" data-delete="${esc(m.id)}">Remove</button>`}</div></div>`).join('');
    if(models.length===1) $('models-list').insertAdjacentHTML('beforeend','<div class="hint-row">To run real deliberations, install a model above and select it for each agent.</div>');
  }catch(e){toast(e.message,true);}
}
async function refreshJobs(){
  try{
    state.jobs=await api('/jobs');
    $('jobs-list').innerHTML=state.jobs.length?state.jobs.slice(0,10).map(j=>`<div class="job-line"><span class="job-state ${esc(j.status)}">${esc(j.status)}</span><strong>${esc(j.type)} · ${esc(j.id.slice(0,8))}</strong><small>${esc(j.message)}</small></div>`).join(''):'<div class="muted">No recent activity.</div>';
    const done=state.jobs.some(j=>j.status==='completed'&&!j._ack);
    if(done){state.jobs.forEach(j=>{if(j.status==='completed')j._ack=true;});}
  }catch(e){console.warn(e);}
}
async function refreshHardware(){
  try{
    const h=await api('/hardware');state.hardware=h;
    const gpu=h.gpu.length?`${h.gpu.map(g=>g.name+' ('+g.vram_gb+' GB)').join(', ')}`:'CPU / no CUDA GPU detected';
    const disk=h.storage?.python_environment;
    $('system-info').textContent=`${h.ram_available_gb} / ${h.ram_total_gb} GB RAM available · ${gpu}${disk?.free_gb!==null&&disk?.free_gb!==undefined?' · '+disk.free_gb+' GB free on app drive':''}`;
    let he=await api('/health');$('cache-limit').value=String(he.max_loaded);
    $('backend-status').textContent='Inference engine online';
    const configured=h.managed_runtimes||{};
    const active=Object.keys(configured).filter(k=>configured[k]);
    const profile=h.accelerator||{};
    $('auto-runtime-status').textContent=(profile.reason||'')+' '+(active.length?'Installed: '+active.join(', '):'No managed inference backend installed yet.');
    $('auto-runtime-button').textContent=active.length?'✓ GPU inference ready · Check/repair setup':'⚡ Auto-configure GPU inference';
    $('advanced-runtime-options').style.display=h.desktop?'none':'block';
  }catch(e){$('backend-status').textContent='Backend unavailable';toast(e.message,true);}
}
async function inspectRepo(){
  const repo=$('repo-input').value.trim();if(!repo){toast('Enter a Hugging Face repository ID.',true);return;}
  $('inspect-button').disabled=true;$('inspect-button').textContent='Inspecting…';
  try{
    const info=await api('/models/inspect','POST',{repo,token:$('hf-token').value||null});
    state.inspected=info;
    const hasGGUF=info.gguf_files.length>0, hasSafe=info.safetensors>0;
    const options=[...(hasGGUF?['gguf']:[]),...(hasSafe?['transformers']:[])];
    $('inspect-panel').classList.remove('hidden');
    $('inspect-panel').innerHTML=`<div class="inspection"><div class="inspect-header"><strong>${esc(info.repo)}</strong><span style="color:var(--accent);font-size:10px">${info.gated?'GATED':info.private?'PRIVATE':'PUBLIC'}</span></div>
      <div class="inspect-stat"><div><span>MODEL TYPE</span><strong>${esc(info.pipeline||'Unspecified')}</strong></div><div><span>ARCHITECTURE</span><strong>${esc(info.architectures.join(', ')||'Unknown')}</strong></div><div><span>GGUF FILES</span><strong>${info.gguf_files.length}</strong></div><div><span>SAFETENSORS</span><strong>${info.safetensors} weight files</strong></div></div>
      ${options.length?`<div class="install-fields"><div class="full"><label class="field-label" for="install-backend">INFERENCE BACKEND</label><select class="select-field" id="install-backend">${options.map(x=>`<option value="${x}" ${x==='gguf'?'selected':''}>${x==='gguf'?'GGUF · llama.cpp':'Safetensors · Transformers'}</option>`).join('')}</select></div><div class="full" id="gguf-field"><label class="field-label" for="gguf-file">CHECKPOINT FILE</label><select class="select-field" id="gguf-file">${info.gguf_files.map(f=>`<option value="${esc(f)}" ${/Q4[_-]K[_-]M/i.test(f)?'selected':''}>${esc(f)}</option>`).join('')}</select></div><div><label class="field-label" for="context-input">CONTEXT TOKENS</label><input id="context-input" type="number" class="text-field" min="512" max="131072" value="4096" /></div><div><label class="field-label" for="gpu-layers">GGUF GPU LAYERS</label><input id="gpu-layers" type="number" class="text-field" min="-1" max="1000" value="-1" title="-1 = all layers possible"/></div><div class="full" id="trust-field" style="display:none"><label class="trust-label"><input type="checkbox" id="trust-code" /> Allow custom repository Python code (unsafe: may run arbitrary code)</label></div></div>
      <button class="primary-button" id="install-button" style="margin-top:18px">Download to local library <span>↓</span></button>`:'<p class="field-help">No GGUF or safetensors causal-LM weights were found. This repository may not be runnable with the included backends.</p>'}
      <p class="field-help">Commit ${esc(info.revision.slice(0,12))}. Only install models from publishers you trust; inspect their licenses and model cards.</p></div>`;
    if(options.length){$('install-backend').addEventListener('change',updateBackendFields);updateBackendFields();$('install-button').addEventListener('click',installRepo);}
  }catch(e){state.inspected=null;toast(e.message,true);}
  finally{$('inspect-button').disabled=false;$('inspect-button').textContent='Inspect ↗';}
}
function updateBackendFields(){const gguf=$('install-backend').value==='gguf';$('gguf-field').style.display=gguf?'':'none';$('trust-field').style.display=gguf?'none':'block';$('gpu-layers').disabled=!gguf;}
async function installRepo(){
  if(!state.inspected)return;
  const backend=$('install-backend').value;
  if($('trust-code')?.checked && !window.confirm('This permits executing Python code from the model repository on your computer. Only proceed if you trust the publisher and have reviewed the repository. Continue?'))return;
  $('install-button').disabled=true;
  try{
    const data={repo:state.inspected.repo,revision:state.inspected.revision,backend,
      filename:backend==='gguf'?$('gguf-file').value:undefined,token:$('hf-token').value||null,
      context:Number($('context-input').value),gpu_layers:Number($('gpu-layers').value),
      trust_remote_code:!!$('trust-code')?.checked};
    const res=await api('/models/install','POST',data);
    toast('Download started · activity '+res.job_id.slice(0,8));refreshJobs();
  }catch(e){toast(e.message,true);}finally{$('install-button').disabled=false;}
}
async function addEndpoint(){
  try{
    await api('/models/endpoint','POST',{base_url:$('endpoint-url').value,model_name:$('endpoint-model').value});
    $('endpoint-model').value='';toast('Inference endpoint registered.');refreshModels();
  }catch(e){toast(e.message,true);}
}
function updateSpecCount(){$('spec-count').textContent=$('spec-input').value.length.toLocaleString()+' characters';}
async function startCouncil(){
  if(state.active){toast('A council is already running.',true);return;}
  const spec=$('spec-input').value.trim();
  if(!spec){toast('Paste or import a specification first.',true);return;}
  if(spec.length>500000){toast('Specification exceeds the 500,000-character limit.',true);return;}
  if(state.agents.length<2){toast('At least two agents are required.',true);return;}
  const body={title:$('council-title').value.trim(),spec,agents:state.agents,
    chair_id:state.chair_id,rounds:Number($('rounds').value),
    input_budget:Number($('budget-select').value),
    response_detail:$('detail-select').value,auto_continue:$('auto-continue').checked,
    max_continuations:Number($('continuations-select').value),criteria:$('criteria').value.split('\n').map(x=>x.trim()).filter(Boolean)};
  $('run-button').disabled=true;
  try{const res=await api('/councils','POST',body);state.run={id:res.id};state.active=true;
    $('active-indicator').classList.remove('hidden');toast('Council convened. Transcript will update as agents respond.');navigate('results');
  }catch(e){toast(e.message,true);}finally{$('run-button').disabled=false;}
}
async function refreshRun(){
  if(!state.run?.id)return;
  try{
    const run=await api('/councils/'+state.run.id);
    state.run=run;state.active=run.status==='running'||run.status==='queued';
    $('active-indicator').classList.toggle('hidden',!state.active);
    $('run-empty').classList.add('hidden');$('run-content').classList.remove('hidden');
    $('run-title').textContent=run.title;
    $('run-subtitle').textContent=`${run.agents.length} members · ${run.rounds} rounds · ${run.messages.length} responses · ${run.response_detail||'legacy'} detail · ${run.phase}`;
    $('run-status').className='session-status '+run.status;
    $('run-status').textContent=run.status;
    $('cancel-run').classList.toggle('hidden',!state.active);
    $('export-md').disabled=false;$('export-json').disabled=false;
    const alerts=run.warnings.map(w=>`<div class="warning-banner">⚠ ${esc(w)}</div>`).join('')+(run.error?`<div class="warning-banner error-banner">${esc(run.error)}</div>`:'');
    $('run-alerts').innerHTML=alerts;
    const messageKey=run.id+':'+run.messages.length;
    if(messageKey!==state.lastMessageKey){state.lastMessageKey=messageKey;renderTranscript(run);}
    const reportKey=run.id+':'+run.status;
    if(reportKey!==state.lastReportKey){state.lastReportKey=reportKey;renderReport(run);}
  }catch(e){toast(e.message,true);}
}
function completionNote(m){
  if(!m.completion_status)return '';
  const labels={complete:'Completed',partial:'Potentially cut off',unknown:'Finish unknown'};
  const css=m.completion_status==='partial'?'partial':m.completion_status==='complete'?'complete':'unknown';
  const calls=m.generation_calls||1;
  const title=[m.completion_warning, `Backend stop: ${m.finish_reason||'unknown'}`, `Generation calls: ${calls}`].filter(Boolean).join(' · ');
  return `<span class="generation-pill ${css}" title="${esc(title)}">${esc(labels[m.completion_status]||m.completion_status)}${calls>1?' · '+calls+' parts':''}</span>`;
}
function renderTranscript(run){
  $('message-number').textContent=run.messages.length+' message'+(run.messages.length===1?'':'s');
  $('transcript-zone').innerHTML=run.messages.length?run.messages.map(m=>`<article class="message-card" id="msg-${esc(m.id)}"><div class="message-header"><div class="message-avatar">${esc(m.agent.slice(0,2).toUpperCase())}</div><div style="min-width:0"><strong>${esc(m.agent)}</strong><p>${esc(m.phase)} · ${esc(m.model_name)}</p></div>${completionNote(m)}<span class="message-id">${esc(m.id)}</span></div><div class="message-text">${esc(m.text)}</div>${m.completion_warning?`<div class="generation-warning">${esc(m.completion_warning)}</div>`:``}<button class="message-toggle">Read full response ↓</button></article>`).join(''):`<div class="panel waiting">${run.status==='failed'?'No responses recorded.':'<div class="spinner"></div><strong>Preparing the first agent…</strong><p>Model loading and inference may take time on local hardware.</p>'}</div>`;
  document.querySelectorAll('.message-toggle').forEach(btn=>btn.addEventListener('click',()=>{const card=btn.closest('.message-card');card.classList.toggle('expanded');btn.textContent=card.classList.contains('expanded')?'Collapse ↑':'Read full response ↓';}));
}
function renderReport(run){
  const r=run.report;
  if(!r){$('report-zone').innerHTML=run.status==='failed'?'<div class="panel waiting"><strong>Deliberation failed</strong><p>Review the error above and any recorded model responses.</p></div>':run.status==='cancelled'?'<div class="panel waiting"><strong>Session cancelled</strong><p>The partial transcript remains available.</p></div>':'<div class="panel waiting"><div class="spinner"></div><strong>Deliberation in progress</strong><p>Findings appear after the chair completes synthesis.</p></div>';return;}
  const validRefs=new Set(run.messages.map(m=>m.id));
  let findings=(Array.isArray(r.findings)?r.findings:[]);
  let html=`<div class="report-card"><h3>Executive summary</h3><p>${esc(r.executive_summary||'No summary returned.')}</p></div>`;
  html+=`<div class="report-card"><h3>Prioritized findings <span class="muted">(${findings.length})</span></h3>${findings.length?findings.map((f,i)=>`<div class="finding"><div class="finding-top"><h4>${i+1}. ${esc(f.title)}</h4><span class="severity ${esc(String(f.severity||'').toLowerCase())}">${esc(f.severity||'unrated')}</span></div><p>${esc(f.rationale||'')}</p><p class="recommendation">${esc(f.recommendation||'')}</p><div>${(Array.isArray(f.sources)?f.sources:[]).filter(id=>validRefs.has(id)).map(id=>`<button class="source-ref" data-source="${esc(id)}">↗ ${esc(id)}</button>`).join('')}</div></div>`).join(''):'<p>No structured findings were returned. Check the raw synthesis message in proceedings.</p>'}</div>`;
  for(const [title,key] of [['Areas of agreement','agreement'],['Unresolved dissent','dissent'],['Open questions','open_questions'],['Recommended next steps','next_steps']]){
    const items=Array.isArray(r[key])?r[key]:[];
    html+=`<div class="report-card"><h3>${title}</h3>${items.length?`<ul class="report-list">${items.map(x=>`<li>${esc(typeof x==='string'?x:JSON.stringify(x))}</li>`).join('')}</ul>`:'<p>None recorded.</p>'}</div>`;
  }
  if(r.parse_warning)html+=`<div class="warning-banner">${esc(r.parse_warning)}</div>`;
  $('report-zone').innerHTML=html;
  document.querySelectorAll('[data-source]').forEach(btn=>btn.addEventListener('click',()=>{const target=$('msg-'+btn.dataset.source);if(target){target.classList.add('expanded');target.querySelector('.message-toggle').textContent='Collapse ↑';target.scrollIntoView({behavior:'smooth',block:'center'});}}));
}
async function refreshArchives(){
  try{
    state.sessions=await api('/councils');
    $('archive-list').innerHTML=state.sessions.length?state.sessions.map(r=>`<div class="archive-item"><button data-open="${esc(r.id)}">${esc(r.title)}<small>${esc(r.phase||'')}&nbsp; · &nbsp;${esc(r.id)}</small></button><span class="archive-status">${esc(r.status)}</span><span>${esc((r.updated_at||'').replace('T',' ').slice(0,16))} UTC</span></div>`).join(''):'<div class="archive-empty">No previous sessions yet.</div>';
  }catch(e){toast(e.message,true);}
}
async function start(){
  document.querySelectorAll('[data-nav]').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.nav)));
  document.querySelectorAll('[data-go]').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.go)));
  $('add-agent').addEventListener('click',()=>{if(state.agents.length>=12){toast('Maximum 12 agents.',true);return;}state.agents.push({id:uid(),name:'New evaluator',role:'Independent technical reviewer.',model_id:'demo',temperature:.45});renderAgents();});
  $('agents-list').addEventListener('click',event=>{let btn=event.target.closest('[data-remove]');if(!btn)return;if(state.agents.length<=2){toast('A council needs at least two members.',true);return;}state.agents=state.agents.filter(a=>a.id!==btn.dataset.remove);if(!state.agents.some(a=>a.id===state.chair_id))state.chair_id=state.agents[0].id;renderAgents();});
  $('agents-list').addEventListener('input',event=>{const el=event.target;const a=state.agents.find(a=>a.id===el.closest('[data-agent]')?.dataset.agent);if(!a||!el.dataset.field)return;const f=el.dataset.field;a[f]=f==='temperature'?Number(el.value):el.value;});
  $('agents-list').addEventListener('change',event=>{const el=event.target;const a=state.agents.find(a=>a.id===el.closest('[data-agent]')?.dataset.agent);if(a&&el.dataset.field){a[el.dataset.field]=el.dataset.field==='temperature'?Number(el.value):el.value;}});
  $('chair-select').addEventListener('change',e=>state.chair_id=e.target.value);
  $('rounds').addEventListener('input',()=>{$('rounds-value').textContent=$('rounds').value+' round'+($('rounds').value==='1'?'':'s');$('run-estimate').textContent=`${state.agents.length} agents · ${$('rounds').value} rounds · ${state.agents.length*(1+2*Number($('rounds').value))+1} baseline calls (continuations may add calls)`;});
  $('auto-continue').addEventListener('change',e=>$('continuations-select').disabled=!e.target.checked);
  $('spec-input').addEventListener('input',updateSpecCount);
  $('upload-button').addEventListener('click',()=>$('spec-file').click());
  $('spec-file').addEventListener('change',async event=>{const f=event.target.files[0];if(!f)return;if(f.size>2000000){toast('Only files under 2 MB can be imported.',true);return;}const content=await f.text();$('spec-input').value=content;updateSpecCount();$('council-title').value=f.name.replace(/\.[^.]+$/,'').slice(0,160);toast('Imported '+f.name);});
  $('run-button').addEventListener('click',startCouncil);
  $('inspect-button').addEventListener('click',inspectRepo);
  $('repo-input').addEventListener('keydown',e=>{if(e.key==='Enter')inspectRepo();});
  $('endpoint-button').addEventListener('click',addEndpoint);
  $('auto-runtime-button').addEventListener('click',async()=>{
    $('auto-runtime-button').disabled=true;
    try{const job=await api('/runtimes/auto','POST');toast('Hardware detection and inference setup started · '+job.job_id.slice(0,8));refreshJobs();}
    catch(e){toast(e.message,true);}finally{$('auto-runtime-button').disabled=false;}
  });
  document.querySelectorAll('[data-runtime]').forEach(btn=>btn.addEventListener('click',async()=>{
    const kind=btn.dataset.runtime;
    const description=kind==='transformers-cpu'?'CPU-only PyTorch, smaller download, no GPU acceleration':kind==='transformers'?'standard PyTorch, potentially many GB of downloads and temporary storage':'GGUF llama.cpp backend (may require compiler)';
    if(!window.confirm(`Install ${description} in this Python environment? Continue?`))return;
    try{const job=await api('/runtimes/install','POST',{backend:kind});toast(`Runtime installation queued · ${job.job_id.slice(0,8)}`);refreshJobs();}catch(e){toast(e.message,true);}
  }));
  $('cache-limit').addEventListener('change',async e=>{try{await api('/models/cache','PUT',{max_loaded:Number(e.target.value)});toast('Resident model limit updated.');}catch(err){toast(err.message,true);}});
  $('models-list').addEventListener('click',async event=>{
    const load=event.target.closest('[data-load]');const del=event.target.closest('[data-delete]');
    try{if(load){const res=await api('/models/'+load.dataset.load+'/load','POST');toast('Model load started · '+res.job_id.slice(0,8));refreshJobs();}
      if(del){const model=state.models.find(m=>m.id===del.dataset.delete);if(!window.confirm('Remove '+(model?.label||'this model')+' from the library? Its app-managed model files will be deleted.'))return;await api('/models/'+del.dataset.delete,'DELETE');toast('Model removed from the library.');refreshModels();}}
    catch(e){toast(e.message,true);}
  });
  $('cancel-run').addEventListener('click',async()=>{if(!state.run)return;try{await api('/councils/'+state.run.id+'/cancel','POST');toast('Cancellation requested; current inference finishes first.');}catch(e){toast(e.message,true);}});
  $('export-md').addEventListener('click',()=>{if(state.run)location.href='/api/councils/'+state.run.id+'/export?format=md';});
  $('export-json').addEventListener('click',()=>{if(state.run)location.href='/api/councils/'+state.run.id+'/export?format=json';});
  $('refresh-archives').addEventListener('click',refreshArchives);
  $('archive-list').addEventListener('click',event=>{let btn=event.target.closest('[data-open]');if(btn){state.run={id:btn.dataset.open};state.lastMessageKey='';state.lastReportKey='';navigate('results');}});
  updateSpecCount();renderAgents();
  await Promise.all([refreshModels(),refreshHardware(),refreshArchives()]);
  // Resume active session after tab refresh (for single-user local server).
  try{let health=await api('/health');if(health.active_run){state.run={id:health.active_run};state.active=true;state.lastMessageKey='';state.lastReportKey='';navigate('results');}}catch(e){}
  setInterval(async()=>{if(state.run?.id)await refreshRun();if(state.page==='models'||state.jobs.some(j=>['queued','running'].includes(j.status))){await refreshJobs();await refreshModels();if(state.page==='models')await refreshHardware();}},2300);
}
document.addEventListener('DOMContentLoaded',start);
