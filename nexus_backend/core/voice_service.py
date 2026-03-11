"""
VoiceService — STT via Deepgram, TTS via ElevenLabs.

Both services are called via httpx (already a project dependency).
No extra SDKs required.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger("voice_service")


class VoiceService:
    def __init__(
        self,
        deepgram_key: str,
        elevenlabs_key: str,
        elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    ) -> None:
        self._deepgram_key = deepgram_key
        self._elevenlabs_key = elevenlabs_key
        self._voice_id = elevenlabs_voice_id

    async def transcribe(self, audio_bytes: bytes, mimetype: str = "audio/webm") -> str:
        """Send audio bytes to Deepgram and return the transcript string."""
        import httpx

        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true"
        headers = {
            "Authorization": f"Token {self._deepgram_key}",
            "Content-Type": mimetype,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=audio_bytes, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        try:
            transcript = (
                data["results"]["channels"][0]["alternatives"][0]["transcript"]
            )
        except (KeyError, IndexError):
            transcript = ""

        log.info("voice_service.transcribed", chars=len(transcript))
        return transcript.strip()

    async def synthesize(self, text: str) -> bytes:
        """Send text to ElevenLabs and return raw audio/mpeg bytes."""
        import httpx

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
        headers = {
            "xi-api-key": self._elevenlabs_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            audio = resp.content

        log.info("voice_service.synthesized", bytes=len(audio))
        return audio
