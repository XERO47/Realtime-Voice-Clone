import { VoxRTC } from './rtc-core.js';

const $ = id => document.getElementById(id);
const rtc = new VoxRTC($('attackRemoteAudio'));
let selectedSource = 'tts';
let generatedAudio = null;
let generatedUrl = null;
let cloneAudio = null;
let cloneUrl = null;
let cloneName = null;
let connected = false;
let online = false;
let transmissionTimer = null;

class SyntheticAudioBus {
  constructor() {
    this.context = null;
    this.destination = null;
    this.silence = null;
    this.activeSource = null;
  }

  async ensure() {
    const liveTrack = this.destination?.stream.getAudioTracks()[0];
    if (this.context && liveTrack?.readyState === 'live') {
      await this.context.resume();
      return this.destination.stream;
    }
    if (this.context) await this.context.close().catch(() => {});
    this.context = new AudioContext();
    this.destination = this.context.createMediaStreamDestination();
    const silentGain = this.context.createGain();
    silentGain.gain.value = 0;
    this.silence = this.context.createConstantSource();
    this.silence.connect(silentGain).connect(this.destination);
    this.silence.start();
    await this.context.resume();
    return this.destination.stream;
  }

  async transmit(arrayBuffer) {
    await this.ensure();
    if (this.activeSource) {
      try { this.activeSource.stop(); } catch (_) {}
    }
    const decoded = await this.context.decodeAudioData(arrayBuffer.slice(0));
    const source = this.context.createBufferSource();
    source.buffer = decoded;
    source.connect(this.destination);
    source.start();
    this.activeSource = source;
    source.onended = () => { if (this.activeSource === source) this.activeSource = null; };
    return decoded.duration;
  }
}

const syntheticBus = new SyntheticAudioBus();

function normalize(value) { return value.trim().toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 20); }
function attackerId() { return `REDTEAM-${Math.floor(1000 + Math.random() * 9000)}`; }
function toast(message) { const element=$('toast');element.textContent=message;element.classList.add('show');setTimeout(()=>element.classList.remove('show'),2600); }
function stamp() { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
function log(message) { const row=document.createElement('p');row.innerHTML=`<time>${stamp()}</time>${message}`;$('attackLog').prepend(row); }

function updateCharacterCount() { $('characterCount').textContent=`${$('attackScript').value.length} / 500`; }
function updateSourceUI() {
  document.querySelectorAll('[data-source]').forEach(button => button.classList.toggle('selected', button.dataset.source === selectedSource));
  $('ttsWorkbench').classList.toggle('hidden', selectedSource !== 'tts');
  $('cloneWorkbench').classList.toggle('hidden', selectedSource !== 'clone');
  const labels={tts:'Synthetic TTS',mic:'Live microphone',clone:'Voice clone clip'};
  $('activeSourceLabel').textContent=labels[selectedSource] || 'Unavailable';
  $('sourceReadyPill').textContent=selectedSource === 'tts' ? 'TTS READY' : selectedSource === 'clone' ? (cloneAudio ? 'CLONE ARMED' : 'LOAD CLONE') : 'MICROPHONE';
  $('injectSpeech').disabled=!(connected && generatedAudio && selectedSource === 'tts');
  $('injectClone').disabled=!(connected && cloneAudio && selectedSource === 'clone');
}

function setCallState(label, active=false) {
  $('attackCallState').textContent=label;
  $('callStateDot').classList.toggle('live',active);
  $('attackCallActions').classList.toggle('hidden',!active);
}

async function activateSource(mode) {
  try {
    let stream;
    if (mode === 'tts' || mode === 'clone') stream=await syntheticBus.ensure();
    else stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    await rtc.replaceLocalStream(stream);
    selectedSource=mode;
    updateSourceUI();
    const sourceLabel={tts:'synthetic TTS',clone:'authorized voice clone',mic:'live microphone'}[mode];
    log(`Outbound source changed to ${sourceLabel}.`);
  } catch (error) { toast(`Could not activate source: ${error.message}`); }
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g,character=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[character]);
}

async function armClone(blob, name) {
  if (!blob.size) throw new Error('The selected clone clip is empty.');
  if (blob.size > 50 * 1024 * 1024) throw new Error('Clone clips must be 50 MB or smaller.');
  cloneAudio=await blob.arrayBuffer();
  cloneName=name;
  if(cloneUrl)URL.revokeObjectURL(cloneUrl);
  cloneUrl=URL.createObjectURL(blob);
  $('clonePreview').src=cloneUrl;
  $('cloneClipName').textContent=name;
  $('cloneClipSize').textContent=formatBytes(blob.size);
  $('cloneClip').classList.remove('hidden');
  await activateSource('clone');
  updateSourceUI();
  log(`Armed authorized clone payload ${name}.`);
}

async function loadClonerVoices() {
  $('clonerVoiceSelect').innerHTML='<option value="">Checking cloning service...</option>';
  $('generateClonerClip').disabled=true;
  try {
    const status=await fetch('/api/cloner/status').then(response=>response.json());
    if(!status.connected){$('clonerStatusText').textContent='Cloning service offline';$('clonerVoiceSelect').innerHTML='<option value="">Cloning service offline</option>';return;}
    if(!status.loaded){$('clonerStatusText').textContent='Cloning service online — model not loaded';$('clonerVoiceSelect').innerHTML='<option value="">Load the model at :8002</option>';return;}
    const data=await fetch('/api/cloner/voices').then(response=>response.json());
    if(!data.voices.length){$('clonerStatusText').textContent=`Cloning service online — no voices encoded yet`;$('clonerVoiceSelect').innerHTML='<option value="">Encode a voice at :8002</option>';return;}
    $('clonerStatusText').textContent=`Cloning service online — ${data.voices.length} voice(s) encoded`;
    $('clonerVoiceSelect').innerHTML=data.voices.map(voice=>`<option value="${voice.id}">${escapeHtml(voice.name)}</option>`).join('');
    $('generateClonerClip').disabled=false;
    $('clonerGenerationMessage').textContent='Ready';
  } catch(error) {
    $('clonerStatusText').textContent='Cloning service unavailable';
    $('clonerVoiceSelect').innerHTML='<option value="">Cloning service unavailable</option>';
  }
}

$('refreshClonerVoices').onclick=loadClonerVoices;
$('generateClonerClip').onclick=async()=>{
  const voiceId=$('clonerVoiceSelect').value;
  const text=$('clonerScript').value.trim();
  if(!voiceId)return toast('Choose an encoded voice first.');
  if(!text)return toast('Enter a script.');
  if(!$('clonerConsent').checked)return toast('Confirm speaker consent first.');
  $('generateClonerClip').disabled=true;
  $('clonerGenerationMessage').textContent='Generating...';
  try {
    const response=await fetch('/api/cloner/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voice_id:voiceId,text,consent:true})});
    if(!response.ok){const detail=await response.json();throw new Error(detail.detail||'Generation failed.');}
    const blob=await response.blob();
    const label=$('clonerVoiceSelect').selectedOptions[0]?.textContent||'cloned voice';
    await armClone(blob,`Cloner: ${label}`);
    $('clonerGenerationMessage').textContent='Payload generated';
    log(`Generated a cloned payload from the cloning service (${label}).`);
  } catch(error) {
    $('clonerGenerationMessage').textContent='Generation failed';
    toast(error.message);
  } finally { $('generateClonerClip').disabled=false; }
};

async function loadCloneLibrary() {
  $('cloneLibrary').innerHTML='<option value="">Refreshing library...</option>';
  try {
    const response=await fetch('/api/voice-clones');
    if(!response.ok)throw new Error('Could not read the clone library.');
    const data=await response.json();
    $('cloneLibrary').innerHTML='<option value="">Choose a preloaded clip</option>'+data.clips.map(clip=>`<option value="${encodeURIComponent(clip.name)}">${escapeHtml(clip.name)} · ${formatBytes(clip.bytes)}</option>`).join('');
    if(!data.clips.length)$('cloneLibrary').innerHTML='<option value="">No clips found — use local upload</option>';
  } catch(error) {
    $('cloneLibrary').innerHTML='<option value="">Clone library unavailable</option>';
    toast(error.message);
  }
}

async function loadVoices() {
  try {
    const status=await fetch('/api/tts/status').then(response=>response.json());
    if(!status.installed)throw new Error('Local TTS model is not installed.');
    const data=await fetch('/api/tts/voices').then(response=>response.json());
    const preferred=['af_sarah','af_heart','af_nova','af_river','am_adam','am_michael','bf_emma','bm_george'];
    const ordered=[...preferred.filter(voice=>data.voices.includes(voice)),...data.voices.filter(voice=>!preferred.includes(voice))];
    $('attackVoice').innerHTML=ordered.map(voice=>`<option value="${voice}">${voice.replaceAll('_',' ')}</option>`).join('');
  } catch(error) {
    $('generationMessage').textContent='TTS engine unavailable';
    $('generateAttackSpeech').disabled=true;
    toast(error.message);
  }
}

$('attackerId').value=attackerId();
$('attackScript').addEventListener('input',updateCharacterCount);
$('attackSpeed').addEventListener('input',()=>{$('speedValue').textContent=`${Number($('attackSpeed').value).toFixed(2)}x`;});
document.querySelectorAll('[data-source]').forEach(button=>button.addEventListener('click',()=>activateSource(button.dataset.source)));
$('refreshClones').onclick=loadCloneLibrary;
$('cloneLibrary').onchange=async()=>{
  if(!$('cloneLibrary').value)return;
  try {
    const response=await fetch(`/api/voice-clones/${$('cloneLibrary').value}`);
    if(!response.ok)throw new Error('Could not load that clone clip.');
    await armClone(await response.blob(),decodeURIComponent($('cloneLibrary').value));
  } catch(error) { toast(error.message); }
};
$('cloneUpload').onchange=async()=>{
  const file=$('cloneUpload').files?.[0];
  if(!file)return;
  try { await armClone(file,file.name); }
  catch(error) { toast(error.message); }
};

$('goOnlineAttack').onclick=async()=>{
  const id=normalize($('attackerId').value);
  if(!id)return toast('Choose an attacker ID first.');
  $('goOnlineAttack').disabled=true;
  $('attackServiceStatus').querySelector('span').textContent='Connecting';
  try {
    await activateSource(selectedSource);
    await rtc.connect(id,'attacker');
    online=true;
    $('attackerId').disabled=true;
    $('attackServiceStatus').classList.add('online');
    $('attackServiceStatus').querySelector('span').textContent=`Online as ${rtc.userId}`;
    $('startAttackCall').disabled=false;
    setCallState('Ready to dial');
    log(`Registered ${rtc.userId} on the VoIP service.`);
  } catch(error) {
    $('goOnlineAttack').disabled=false;
    $('attackServiceStatus').querySelector('span').textContent='Offline';
    toast(error.message);
  }
};

$('startAttackCall').onclick=async()=>{
  const target=normalize($('attackTarget').value);
  if(!online)return toast('Go online first.');
  if(!target)return toast('Enter the protected user ID.');
  await activateSource(selectedSource);
  rtc.startCall(target);
  $('attackPeer').textContent=target;
  setCallState('Calling target...');
  $('startAttackCall').disabled=true;
  log(`Calling protected endpoint ${target}.`);
};

$('generateAttackSpeech').onclick=async()=>{
  const text=$('attackScript').value.trim();
  if(!text)return toast('Enter an attack script.');
  $('generateAttackSpeech').disabled=true;
  $('generationMessage').textContent='Generating locally...';
  try {
    const response=await fetch('/api/tts/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,voice:$('attackVoice').value,speed:Number($('attackSpeed').value)})});
    if(!response.ok){const detail=await response.json();throw new Error(detail.detail||'Generation failed.');}
    const blob=await response.blob();
    generatedAudio=await blob.arrayBuffer();
    if(generatedUrl)URL.revokeObjectURL(generatedUrl);
    generatedUrl=URL.createObjectURL(blob);
    $('attackPreview').src=generatedUrl;
    $('clipDuration').textContent=`${Number(response.headers.get('X-Audio-Duration')).toFixed(1)}s`;
    $('clipGeneration').textContent=`${Number(response.headers.get('X-Generation-Time')).toFixed(1)}s`;
    $('clipVoice').textContent=response.headers.get('X-TTS-Voice')||$('attackVoice').value;
    $('generatedClip').classList.remove('hidden');
    $('generationMessage').textContent='Payload generated';
    updateSourceUI();
    log(`Generated a ${$('clipDuration').textContent} synthetic payload.`);
  } catch(error) {
    $('generationMessage').textContent='Generation failed';
    toast(error.message);
  } finally { $('generateAttackSpeech').disabled=false; }
};

async function transmitPayload(audio,mode,label) {
  if(!connected)return toast('Connect the VoIP call before transmitting.');
  if(!audio)return toast(mode === 'clone' ? 'Load a clone payload first.' : 'Generate a TTS payload first.');
  await activateSource(mode);
  try {
    const duration=await syntheticBus.transmit(audio);
    $('injectSpeech').disabled=true;$('injectClone').disabled=true;
    $('transmitProgress').classList.remove('hidden');
    $('transmitLabel').textContent=label;
    $('transmitBar').style.transition='none';$('transmitBar').style.width='0%';
    requestAnimationFrame(()=>requestAnimationFrame(()=>{$('transmitBar').style.transition=`width ${duration}s linear`;$('transmitBar').style.width='100%';}));
    const started=Date.now();
    clearInterval(transmissionTimer);
    transmissionTimer=setInterval(()=>{const remaining=Math.max(0,duration-(Date.now()-started)/1000);$('transmitCountdown').textContent=`${remaining.toFixed(1)}s`;if(remaining<=0){clearInterval(transmissionTimer);$('transmitProgress').classList.add('hidden');updateSourceUI();log(`${mode === 'clone' ? 'Clone' : 'Synthetic'} transmission completed.`);}},100);
    log(`${mode === 'clone' ? `Voice clone ${cloneName}` : 'Synthetic payload'} transmitted into the live call.`);
  } catch(error) { updateSourceUI();toast(error.message); }
}

$('injectSpeech').onclick=()=>transmitPayload(generatedAudio,'tts','TRANSMITTING SYNTHETIC TTS');
$('injectClone').onclick=()=>transmitPayload(cloneAudio,'clone','TRANSMITTING VOICE CLONE');

rtc.addEventListener('ringing',event=>{rtc.session=event.detail.session;$('attackSession').textContent=rtc.session.session_id;setCallState('Target is ringing');});
rtc.addEventListener('incoming-call',event=>{$('attackerIncomingFrom').textContent=event.detail.from;$('attackerIncoming').classList.remove('hidden');$('attackPeer').textContent=event.detail.from;});
$('attackerAccept').onclick=async()=>{await activateSource(selectedSource);$('attackerIncoming').classList.add('hidden');await rtc.acceptCall();setCallState('Connecting...');};
$('attackerDecline').onclick=()=>{rtc.declineCall();$('attackerIncoming').classList.add('hidden');setCallState('Ready to dial');};
rtc.addEventListener('connecting',()=>setCallState('Negotiating audio...'));
rtc.addEventListener('connected',()=>{connected=true;$('attackSession').textContent=rtc.session?.session_id||'--';$('attackPeer').textContent=rtc.peerId||$('attackPeer').textContent;setCallState('Connected to protected user',true);updateSourceUI();log('WebRTC call connected. Synthetic source is armed.');});
rtc.addEventListener('call-error',event=>{connected=false;setCallState('Ready to dial');$('startAttackCall').disabled=false;toast(event.detail.message);log(`Call failed: ${event.detail.message}`);});
rtc.addEventListener('call-ended',event=>{connected=false;setCallState('Ready to dial');$('startAttackCall').disabled=false;$('attackSession').textContent='--';updateSourceUI();toast(event.detail.reason||'Call ended.');log('Call session ended.');});
rtc.addEventListener('intervention',event=>{const labels={verify:'Identity verification requested by monitoring.',hold:'The call was placed on verification hold.',end:'Monitoring ended this session.'};$('attackIntervention').textContent=labels[event.detail.action]||'Monitoring intervention received.';$('attackIntervention').classList.remove('hidden');setTimeout(()=>$('attackIntervention').classList.add('hidden'),5000);log(labels[event.detail.action]||'Monitoring intervention received.');});
rtc.addEventListener('error',event=>toast(event.detail.message));

$('attackMute').onclick=()=>{const muted=rtc.toggleMute();$('attackMute').textContent=muted?'Unmute':'Mute';log(muted?'Outbound audio muted.':'Outbound audio unmuted.');};
$('attackSpeaker').onclick=()=>{const muted=rtc.toggleSpeaker();$('attackSpeaker').textContent=muted?'Speaker on':'Speaker off';};
$('attackEnd').onclick=()=>{rtc.endCall();connected=false;setCallState('Ready to dial');$('startAttackCall').disabled=false;updateSourceUI();log('Call ended from attack console.');};

updateCharacterCount();
updateSourceUI();
loadVoices();
loadCloneLibrary();
loadClonerVoices();
