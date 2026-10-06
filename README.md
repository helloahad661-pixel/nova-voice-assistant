# nova-voice-assistant

Push-to-talk voice assistant loop for macOS (Apple Silicon): microphone in,
faster-whisper speech-to-text, reply routed through an AI orchestrator,
kokoro speech out.

```bash
pip install -r requirements.txt
python nova_voice_loop.py
```

Runs as a macOS LaunchAgent (`com.nova.voice`) for always-on use. The reply
routing plugs into the [Nova](https://github.com/helloahad661-pixel/Nova)
orchestrator. Needs a mic, ~8 GB RAM, and the whisper/kokoro models on first
run.

Maintained in Nova by `system_agent_real` (macOS integration). Extracted from
the Nova monorepo as a portfolio piece.
