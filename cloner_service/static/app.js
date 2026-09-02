const $=id=>document.getElementById(id);
let selectedVoice=null;
let outputUrl=null;
let poller=null;

function toast(message,error=false){const el=$('toast');el.textContent=message;el.className=`toast show${error?' error':''}`;clearTimeout(el._timer);el._timer=setTimeout(()=>el.className='toast',3600)}
async function errorMessage(response){try{const body=await response.json();return body.detail||JSON.stringify(body)}catch{return `${response.status} ${response.statusText}`}}
function setWorking(working){for(const id of ['loadModel','unloadModel','encodeVoice','generateClone','clearCache'])$(id).disabled=working}

async function refreshStatus(){
  try{
    const status=await (await fetch('/api/status')).json();
    $('healthPill').textContent=status.busy?'BUSY':status.loaded?'MODEL LIVE':'MODEL OFF';
    $('healthPill').className=`pill ${status.busy?'busy':status.loaded?'live':'idle'}`;
    $('operationText').textContent=status.operation||'Waiting';
    $('activeDevice').textContent=status.loaded?(status.device||'—').toUpperCase():'—';
    $('gpuName').textContent=status.cuda_available?(status.gpu_name||'CUDA available')+(status.loaded&&status.device==='cuda'?` · ${status.gpu_allocated_mb} MB`:''):'CPU-only PyTorch build';
    $('voiceMemory').textContent=`${status.cached_voices} / ${status.voices}`;
    $('runtimeNotice').textContent=status.last_error||(!status.cuda_available?'CUDA is not available in this local environment; Auto will use CPU. Install a CUDA PyTorch build on the GPU host.':'CUDA is available. Auto will keep the model on GPU.');
    $('runtimeNotice').className=`notice${status.last_error?' error':''}`;
    setWorking(status.busy);
  }catch{$('healthPill').textContent='SERVICE ERROR';$('healthPill').className='pill idle'}
}

async function refreshVoices(){
  const response=await fetch('/api/voices');
  const data=await response.json();
  const voices=data.voices||[];
  if(selectedVoice&&!voices.some(v=>v.id===selectedVoice))selectedVoice=null;
  if(!selectedVoice&&voices.length)selectedVoice=voices[0].id;
  $('voiceCount').textContent=`${voices.length} profile${voices.length===1?'':'s'}`;
  $('voiceList').innerHTML=voices.length?voices.map(v=>`<div class="voice-item ${v.id===selectedVoice?'selected':''}" data-id="${v.id}"><span class="voice-avatar">${v.name.slice(0,2).toUpperCase()}</span><div><b>${escapeHtml(v.name)}</b><small>${v.duration_seconds}s · ${v.cached?'in memory':'reference saved'}</small></div><i class="cache-dot ${v.cached?'live':''}"></i><button data-delete="${v.id}" title="Delete">×</button></div>`).join(''):'<div class="empty">No voice loaded yet.</div>';
  document.querySelectorAll('.voice-item').forEach(el=>el.onclick=e=>{if(e.target.dataset.delete)return;selectedVoice=el.dataset.id;refreshVoices()});
  document.querySelectorAll('[data-delete]').forEach(el=>el.onclick=async e=>{e.stopPropagation();if(!confirm('Delete this voice profile and its reference file?'))return;const response=await fetch(`/api/voices/${el.dataset.delete}`,{method:'DELETE'});if(!response.ok)return toast(await errorMessage(response),true);await Promise.all([refreshVoices(),refreshStatus()]);toast('Voice profile deleted.')});
}
function escapeHtml(text){const div=document.createElement('div');div.textContent=text;return div.innerHTML}

$('loadModel').onclick=async()=>{
  setWorking(true);$('operationText').textContent='Loading weights…';
  try{const response=await fetch('/api/model/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({device:$('deviceSelect').value})});if(!response.ok)throw new Error(await errorMessage(response));toast('Chatterbox Nano is resident and ready.')}catch(error){toast(error.message,true)}finally{await refreshStatus();setWorking(false)}
};
$('unloadModel').onclick=async()=>{setWorking(true);try{const response=await fetch('/api/model/unload',{method:'POST'});if(!response.ok)throw new Error(await errorMessage(response));toast('Model and voice conditionings released from memory.');await refreshVoices()}catch(error){toast(error.message,true)}finally{await refreshStatus();setWorking(false)}};
$('clearCache').onclick=async()=>{try{const response=await fetch('/api/voices/clear-cache',{method:'POST'});if(!response.ok)throw new Error(await errorMessage(response));toast('Speaker conditionings cleared; references remain saved.');await Promise.all([refreshVoices(),refreshStatus()])}catch(error){toast(error.message,true)}};
$('referenceAudio').onchange=()=>{$('referenceName').textContent=$('referenceAudio').files?.[0]?.name||'Choose a recording'};
$('encodeVoice').onclick=async()=>{
  const file=$('referenceAudio').files?.[0];if(!file)return toast('Choose a reference recording first.',true);if(!$('voiceConsent').checked)return toast('Confirm the speaker’s consent first.',true);
  const body=new FormData();body.append('reference',file);body.append('name',$('voiceName').value);body.append('consent','true');setWorking(true);$('operationText').textContent='Encoding speaker…';
  try{const response=await fetch('/api/voices',{method:'POST',body});if(!response.ok)throw new Error(await errorMessage(response));const data=await response.json();selectedVoice=data.voice.id;toast(`${data.voice.name} is encoded and resident.`);await Promise.all([refreshVoices(),refreshStatus()])}catch(error){toast(error.message,true)}finally{setWorking(false)}
};

for(const [id,out,digits] of [['temperature','tempValue',2],['topP','topPValue',2],['repetition','repValue',2]])$(id).oninput=()=>$(out).textContent=Number($(id).value).toFixed(digits);
$('scriptText').oninput=()=>$('charCount').textContent=$('scriptText').value.length;
$('charCount').textContent=$('scriptText').value.length;
$('generateClone').onclick=async()=>{
  if(!selectedVoice)return toast('Encode or select a voice first.',true);if(!$('generationConsent').checked)return toast('Confirm authorized use before generating.',true);
  setWorking(true);$('generateClone').textContent='Generating on resident model…';
  const request={voice_id:selectedVoice,text:$('scriptText').value,temperature:Number($('temperature').value),top_p:Number($('topP').value),top_k:1000,repetition_penalty:Number($('repetition').value),seed:Number($('seed').value),consent:true};
  try{
    const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});if(!response.ok)throw new Error(await errorMessage(response));
    const metadata=JSON.parse(response.headers.get('X-Clone-Metadata')||'{}');const blob=await response.blob();if(outputUrl)URL.revokeObjectURL(outputUrl);outputUrl=URL.createObjectURL(blob);$('cloneAudio').src=outputUrl;$('downloadClone').href=outputUrl;$('resultName').textContent=`${metadata.voice_name||'Voice'} · ${metadata.device||'device'}`;$('generationTime').textContent=`${metadata.generation_seconds||'—'}s`;$('audioTime').textContent=`${metadata.audio_seconds||'—'}s`;$('rtfValue').textContent=metadata.real_time_factor??'—';$('result').classList.remove('hidden');toast('Cloned speech generated and watermarked.');
  }catch(error){toast(error.message,true)}finally{$('generateClone').textContent='Generate cloned speech';setWorking(false);await refreshStatus()}
};
$('openAttacker').onclick=()=>window.open('http://127.0.0.1:8000/attacker','_blank');

Promise.all([refreshStatus(),refreshVoices()]);poller=setInterval(refreshStatus,2000);

