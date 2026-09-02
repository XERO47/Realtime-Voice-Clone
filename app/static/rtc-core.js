export class VoxRTC extends EventTarget {
  constructor(remoteAudio) {
    super();
    this.remoteAudio = remoteAudio;
    this.signal = null;
    this.peer = null;
    this.localStream = null;
    this.remoteStream = null;
    this.session = null;
    this.userId = null;
    this.peerId = null;
    this.pendingIce = [];
    this.audioSocket = null;
    this.audioContext = null;
    this.worklet = null;
    this.mirrorSource = null;
    this.mirrorSilentGain = null;
    this.workletModuleReady = null;
    this.mirrorSending = false;
    this.mirrorSessionId = null;
    this.selectedMicId = null;
    this.selectedSpeakerId = null;
  }

  async listDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return {
      mics: devices.filter(device => device.kind === 'audioinput'),
      speakers: devices.filter(device => device.kind === 'audiooutput')
    };
  }

  async setMicDevice(deviceId) {
    this.selectedMicId = deviceId || null;
    if (!this.localStream || !this.peer) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: this.micConstraints(), video: false });
    await this.replaceLocalStream(stream);
  }

  async setSpeakerDevice(deviceId) {
    this.selectedSpeakerId = deviceId || null;
    if (typeof this.remoteAudio.setSinkId === 'function') {
      await this.remoteAudio.setSinkId(this.selectedSpeakerId || '');
    }
  }

  micConstraints() {
    const base = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
    return this.selectedMicId ? { ...base, deviceId: { exact: this.selectedMicId } } : base;
  }

  emit(type, detail = {}) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  wsUrl(path) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${location.host}${path}`;
  }

  connect(userId, role = 'consumer') {
    return new Promise((resolve, reject) => {
      this.userId = userId;
      this.signal = new WebSocket(this.wsUrl(`/ws/signaling/${encodeURIComponent(userId)}?role=${encodeURIComponent(role)}`));
      this.signal.onopen = () => this.emit('socket-open');
      this.signal.onerror = () => reject(new Error('Could not connect to the calling service.'));
      this.signal.onclose = event => {
        this.emit('socket-close', { event });
        if (!event.wasClean && !this.session) reject(new Error('Connection closed.'));
      };
      this.signal.onmessage = async event => {
        const message = JSON.parse(event.data);
        if (message.type === 'registered') {
          this.userId = message.user_id;
          this.emit('registered', message);
          return resolve(message);
        }
        if (message.type === 'register-error') return reject(new Error(message.message));
        try { await this.handleSignal(message); }
        catch (error) { this.emit('error', { message: error.message }); }
      };
    });
  }

  send(payload) {
    if (this.signal?.readyState === WebSocket.OPEN) this.signal.send(JSON.stringify(payload));
  }

  async handleSignal(message) {
    if (message.type === 'incoming-call') {
      this.session = message.session;
      this.peerId = message.from;
      return this.emit('incoming-call', message);
    }
    if (message.type === 'call-ringing') {
      this.session = message.session;
      this.peerId = message.peer_id;
      return this.emit('ringing', message);
    }
    if (message.type === 'call-accepted') {
      this.session = message.session;
      await this.ensureMedia();
      await this.createPeer();
      const offer = await this.peer.createOffer();
      await this.peer.setLocalDescription(offer);
      this.send({ type: 'offer', session_id: this.session.session_id, sdp: this.peer.localDescription });
      return this.emit('connecting', message);
    }
    if (message.type === 'offer') {
      await this.ensureMedia();
      await this.createPeer();
      await this.peer.setRemoteDescription(message.sdp);
      await this.flushIce();
      const answer = await this.peer.createAnswer();
      await this.peer.setLocalDescription(answer);
      this.send({ type: 'answer', session_id: this.session.session_id, sdp: this.peer.localDescription });
      return;
    }
    if (message.type === 'answer') {
      await this.peer.setRemoteDescription(message.sdp);
      await this.flushIce();
      return;
    }
    if (message.type === 'ice-candidate') {
      if (this.peer?.remoteDescription) await this.peer.addIceCandidate(message.candidate);
      else this.pendingIce.push(message.candidate);
      return;
    }
    if (message.type === 'call-error' || message.type === 'call-declined') {
      this.emit('call-error', { message: message.message || 'The call was declined.' });
      return this.resetCall();
    }
    if (message.type === 'call-ended') {
      this.emit('call-ended', message);
      return this.resetCall();
    }
    if (message.type === 'intervention') return this.emit('intervention', message);
  }

  startCall(target) {
    this.peerId = target;
    this.prepareAudioMirrorContext().catch(() => {});
    this.send({ type: 'call-request', target });
  }

  async acceptCall() {
    await this.prepareAudioMirrorContext();
    await this.ensureMedia();
    this.send({ type: 'call-accept', session_id: this.session.session_id });
    this.emit('connecting', { by: this.userId });
  }

  declineCall() {
    if (this.session) this.send({ type: 'call-decline', session_id: this.session.session_id });
    this.resetCall();
  }

  async ensureMedia() {
    if (this.localStream?.getAudioTracks().some(track => track.readyState === 'live')) return this.localStream;
    if (this.session) await this.prepareAudioMirrorContext();
    this.localStream = await navigator.mediaDevices.getUserMedia({
      audio: this.micConstraints(),
      video: false
    });
    this.emit('local-stream', { stream: this.localStream });
    await this.startAudioMirror();
    return this.localStream;
  }

  async prepareAudioMirrorContext() {
    if (this.audioContext && this.audioContext.state !== 'closed') {
      if (this.audioContext.state === 'suspended') await this.audioContext.resume().catch(() => {});
      if (this.workletModuleReady) await this.workletModuleReady;
      return this.audioContext;
    }
    this.audioContext = new AudioContext({ latencyHint: 'interactive' });
    this.workletModuleReady = this.audioContext.audioWorklet.addModule('/static/pcm-worklet.js');
    await this.workletModuleReady;
    await this.audioContext.resume().catch(() => {});
    this.audioContext.onstatechange = () => this.emit('mirror-status', {
      state: this.audioContext?.state || 'closed',
      message: this.audioContext?.state === 'running' ? 'PCM processor running' : 'PCM processor suspended'
    });
    return this.audioContext;
  }

  async replaceLocalStream(stream) {
    const track = stream?.getAudioTracks()[0];
    if (!track) throw new Error('The selected source has no audio track.');
    const previous = this.localStream;
    this.localStream = stream;
    const sender = this.peer?.getSenders().find(item => item.track?.kind === 'audio');
    if (sender) await sender.replaceTrack(track);
    if (previous && previous !== stream) previous.getTracks().forEach(item => item.stop());
    if (this.session) {
      await this.stopAudioMirror();
      await this.startAudioMirror();
    }
    this.emit('local-stream', { stream: this.localStream });
    return this.localStream;
  }

  async createPeer() {
    if (this.peer) return this.peer;
    this.peer = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });
    this.localStream.getTracks().forEach(track => this.peer.addTrack(track, this.localStream));
    this.peer.onicecandidate = event => {
      if (event.candidate) this.send({ type: 'ice-candidate', session_id: this.session.session_id, candidate: event.candidate });
    };
    this.peer.ontrack = event => {
      this.remoteStream = event.streams[0];
      this.remoteAudio.srcObject = this.remoteStream;
      this.remoteAudio.play().catch(() => {});
      this.emit('remote-stream', { stream: this.remoteStream });
    };
    this.peer.onconnectionstatechange = () => {
      const state = this.peer?.connectionState;
      this.emit('connection-state', { state });
      if (state === 'connected') {
        this.send({ type: 'call-connected', session_id: this.session.session_id });
        this.emit('connected', { session: this.session });
      }
      if (['failed', 'closed'].includes(state)) this.emit('connection-failed', { state });
    };
    return this.peer;
  }

  async flushIce() {
    while (this.pendingIce.length && this.peer?.remoteDescription) {
      await this.peer.addIceCandidate(this.pendingIce.shift());
    }
  }

  async startAudioMirror() {
    if (!this.localStream || !this.session || this.mirrorSessionId === this.session.session_id) return;
    if (this.worklet || this.audioSocket || this.mirrorSessionId) await this.stopAudioMirror();
    await this.prepareAudioMirrorContext();
    this.mirrorSource = this.audioContext.createMediaStreamSource(this.localStream);
    this.worklet = new AudioWorkletNode(this.audioContext, 'pcm-collector');
    this.mirrorSilentGain = this.audioContext.createGain();
    this.mirrorSilentGain.gain.value = 0;
    this.mirrorSource.connect(this.worklet).connect(this.mirrorSilentGain).connect(this.audioContext.destination);
    this.audioSocket = new WebSocket(this.wsUrl(`/ws/audio/${this.session.session_id}/${encodeURIComponent(this.userId)}?sample_rate=${this.audioContext.sampleRate}`));
    this.audioSocket.binaryType = 'arraybuffer';
    this.audioSocket.onopen = () => this.emit('mirror-status', { state: 'connected', message: 'PCM socket connected; waiting for frames' });
    this.audioSocket.onerror = () => this.emit('mirror-status', { state: 'error', message: 'PCM mirror connection failed' });
    this.audioSocket.onclose = () => this.emit('mirror-status', { state: 'closed', message: 'PCM mirror disconnected' });
    this.worklet.port.onmessage = event => {
      const samples = event.data;
      let energy = 0;
      for (let index=0; index<samples.length; index++) energy += samples[index] * samples[index];
      const level = Math.min(1, Math.sqrt(energy / Math.max(1, samples.length)) * 4);
      this.emit('local-level', { level, active: level >= 0.015 });
      if (this.audioSocket?.readyState === WebSocket.OPEN) {
        this.audioSocket.send(samples.buffer);
        if (!this.mirrorSending) {
          this.mirrorSending = true;
          this.emit('mirror-status', { state: 'streaming', message: 'PCM frames reaching monitor' });
        }
      }
    };
    if (this.audioContext.state === 'suspended') await this.audioContext.resume().catch(() => {});
    this.mirrorSessionId = this.session.session_id;
  }

  async stopAudioMirror() {
    this.mirrorSource?.disconnect();
    this.mirrorSource = null;
    this.worklet?.disconnect();
    this.worklet = null;
    this.mirrorSilentGain?.disconnect();
    this.mirrorSilentGain = null;
    if (this.audioSocket && this.audioSocket.readyState < WebSocket.CLOSING) this.audioSocket.close();
    this.audioSocket = null;
    if (this.audioContext) await this.audioContext.close().catch(() => {});
    this.audioContext = null;
    this.workletModuleReady = null;
    this.mirrorSending = false;
    this.mirrorSessionId = null;
  }

  toggleMute() {
    const track = this.localStream?.getAudioTracks()[0];
    if (!track) return false;
    track.enabled = !track.enabled;
    return !track.enabled;
  }

  toggleSpeaker() {
    this.remoteAudio.muted = !this.remoteAudio.muted;
    return this.remoteAudio.muted;
  }

  endCall() {
    if (this.session) this.send({ type: 'call-end', session_id: this.session.session_id });
    this.resetCall();
  }

  async resetCall() {
    await this.stopAudioMirror();
    this.peer?.close();
    this.peer = null;
    this.localStream?.getTracks().forEach(track => track.stop());
    this.localStream = null;
    this.remoteStream = null;
    this.remoteAudio.srcObject = null;
    this.session = null;
    this.peerId = null;
    this.pendingIce = [];
  }
}
