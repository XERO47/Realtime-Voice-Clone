import { VoxRTC } from './rtc-core.js';

const $ = id => document.getElementById(id);
const rtc = new VoxRTC($('remoteAudio'));
let startedAt = null;
let timer = null;

function generateId() {
  const words = ['RIVER','CEDAR','NOVA','MANGO','LOTUS','CORAL','EMBER','SKY'];
  return `${words[Math.floor(Math.random()*words.length)]}-${Math.floor(1000+Math.random()*9000)}`;
}

function normalize(value) { return value.trim().toUpperCase().replace(/[^A-Z0-9-]/g,'').slice(0,20); }
function initials(value) { return value.split('-')[0].slice(0,2).toUpperCase(); }
function toast(message) { const el=$('toast');el.textContent=message;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600); }
function show(view) { ['setupView','readyView','callView'].forEach(id=>$(id).classList.toggle('hidden',id!==view)); }
function setStatus(label, online=false) { $('serviceStatus').classList.toggle('online',online);$('serviceStatus').querySelector('span').textContent=label; }

function populateSelect(select,items,fallbackLabel){
  const current=select.value;
  select.innerHTML=`<option value="">Default ${fallbackLabel}</option>`+items.map((device,index)=>`<option value="${device.deviceId}">${device.label||`${fallbackLabel} ${index+1}`}</option>`).join('');
  if([...select.options].some(option=>option.value===current))select.value=current;
}

async function refreshDevices(){
  try{
    const{mics,speakers}=await rtc.listDevices();
    [$('micSelect'),$('callMicSelect')].forEach(select=>populateSelect(select,mics,'microphone'));
    const speakerSupported='setSinkId' in HTMLMediaElement.prototype;
    [$('speakerSelect'),$('callSpeakerSelect')].forEach(select=>{
      select.classList.toggle('hidden',!speakerSupported);
      if(speakerSupported)populateSelect(select,speakers,'speaker');
    });
  }catch(_){ /* enumerateDevices can fail before any permission is granted; leave defaults */ }
}

[$('micSelect'),$('callMicSelect')].forEach(select=>select.onchange=async()=>{
  const value=select.value;
  [$('micSelect'),$('callMicSelect')].forEach(other=>{other.value=value;});
  try{await rtc.setMicDevice(value);toast('Microphone switched.');}
  catch(error){toast(`Could not switch microphone: ${error.message}`);}
});
[$('speakerSelect'),$('callSpeakerSelect')].forEach(select=>select.onchange=async()=>{
  const value=select.value;
  [$('speakerSelect'),$('callSpeakerSelect')].forEach(other=>{other.value=value;});
  try{await rtc.setSpeakerDevice(value);toast('Speaker switched.');}
  catch(error){toast(`Could not switch speaker: ${error.message}`);}
});
rtc.addEventListener('local-stream',refreshDevices);
if(navigator.mediaDevices.addEventListener)navigator.mediaDevices.addEventListener('devicechange',refreshDevices);
refreshDevices();

$('identityInput').value=generateId();
$('generateId').onclick=()=>{ $('identityInput').value=generateId(); };
$('goOnline').onclick=async()=>{
  const userId=normalize($('identityInput').value);if(!userId)return toast('Choose a calling ID first.');
  $('goOnline').disabled=true;setStatus('Connecting…');
  try{await rtc.connect(userId);$('displayId').textContent=rtc.userId;$('detailLocal').textContent=rtc.userId;show('readyView');setStatus('Online',true);}
  catch(error){toast(error.message);setStatus('Offline');$('goOnline').disabled=false;}
};
$('copyId').onclick=async()=>{await navigator.clipboard.writeText(rtc.userId);toast('Calling ID copied.');};
$('callButton').onclick=()=>{const target=normalize($('targetInput').value);if(!target)return toast('Enter the other person’s ID.');rtc.startCall(target);showCall(target,'Calling…');};

function showCall(peer,status){show('callView');$('peerName').textContent=peer;$('peerRing').textContent=initials(peer);$('detailRemote').textContent=peer;$('callState').textContent=status;$('callTitle').textContent='Protected audio call';if(rtc.session)$('sessionLabel').textContent=`SESSION ${rtc.session.session_id}`;}
function startTimer(){if(timer)return;startedAt=Date.now();timer=setInterval(()=>{const seconds=Math.floor((Date.now()-startedAt)/1000);$('callTimer').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;},1000);}
function resetUI(message){clearInterval(timer);timer=null;startedAt=null;$('incomingModal').classList.add('hidden');$('interventionBanner').classList.add('hidden');$('localMicLevel').style.width='0%';$('localMicPercent').textContent='0%';$('localMicStatus').textContent='The meter will move only when real PCM frames are captured.';$('detailMirror').textContent='Waiting';show('readyView');if(message)toast(message);}

rtc.addEventListener('ringing',event=>{rtc.session=event.detail.session;showCall(event.detail.peer_id,'Waiting for answer');$('sessionLabel').textContent=`SESSION ${rtc.session.session_id}`;});
rtc.addEventListener('incoming-call',event=>{$('incomingFrom').textContent=event.detail.from;$('incomingModal').classList.remove('hidden');});
$('acceptButton').onclick=async()=>{try{$('incomingModal').classList.add('hidden');showCall(rtc.peerId,'Connecting secure audio…');$('sessionLabel').textContent=`SESSION ${rtc.session.session_id}`;await rtc.acceptCall();}catch(error){toast(`Microphone unavailable: ${error.message}`);rtc.declineCall();resetUI();}};
$('declineButton').onclick=()=>{rtc.declineCall();resetUI('Call declined.');};
rtc.addEventListener('connecting',()=>{$('callState').textContent='Connecting secure audio…';$('detailAudio').textContent='Connecting';});
rtc.addEventListener('connected',()=>{$('callState').textContent='Connected';$('detailAudio').textContent='Checking input';$('peerRing').classList.add('speaking');startTimer();});
rtc.addEventListener('local-level',event=>{const level=Math.round(event.detail.level*100);$('localMicLevel').style.width=`${level}%`;$('localMicPercent').textContent=`${level}%`;$('localMicStatus').textContent=event.detail.active?'Voice activity detected':'PCM streaming · input is quiet';$('detailAudio').textContent=event.detail.active?'Voice detected':'Connected · quiet';});
rtc.addEventListener('mirror-status',event=>{const labels={connected:'Socket connected',streaming:'PCM streaming',running:'Processor running',suspended:'Processor suspended',error:'Mirror error',closed:'Disconnected'};$('detailMirror').textContent=labels[event.detail.state]||event.detail.message;$('detailMirror').classList.toggle('detail-error',['suspended','error','closed'].includes(event.detail.state));});
rtc.addEventListener('call-error',event=>resetUI(event.detail.message));
rtc.addEventListener('call-ended',event=>resetUI(event.detail.reason||'Call ended.'));
rtc.addEventListener('error',event=>toast(event.detail.message));
rtc.addEventListener('intervention',event=>{const action=event.detail.action;const messages={verify:'Monitoring console requested identity verification before continuing.',hold:'This call has been placed on a verification hold.',end:'The monitoring console ended this session.'};$('interventionBanner').textContent=messages[action]||'Verification requested.';$('interventionBanner').classList.remove('hidden');if(action==='end')resetUI(messages.end);});
$('muteButton').onclick=()=>{const muted=rtc.toggleMute();$('muteButton').classList.toggle('active',muted);$('muteButton').title=muted?'Unmute microphone':'Mute microphone';toast(muted?'Microphone muted.':'Microphone on.');};
$('speakerButton').onclick=()=>{const muted=rtc.toggleSpeaker();$('speakerButton').classList.toggle('active',muted);toast(muted?'Speaker silenced.':'Speaker on.');};
$('endButton').onclick=()=>{rtc.endCall();resetUI('Call ended.');};
