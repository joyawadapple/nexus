"""
Voice router — STT and TTS endpoints.

POST /voice/transcribe  — accepts audio file, returns transcript text
POST /voice/synthesize  — accepts text, returns audio/mpeg stream
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.voice_service import VoiceService

log = structlog.get_logger("voice_router")
router = APIRouter(prefix="/voice", tags=["voice"])

_voice_service: VoiceService | None = None


def set_services(voice_service: VoiceService) -> None:
    global _voice_service
    _voice_service = voice_service


class SynthesizeRequest(BaseModel):
    text: str


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio to text via Deepgram."""
    if _voice_service is None:
        raise HTTPException(status_code=503, detail="Voice service not configured")

    audio_bytes = await file.read()
    mimetype = file.content_type or "audio/webm"

    transcript = await _voice_service.transcribe(audio_bytes, mimetype)

    if not transcript:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    log.info("voice.transcribed", length=len(transcript))
    return {"transcript": transcript}


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Convert text to speech via ElevenLabs and stream audio/mpeg."""
    if _voice_service is None:
        raise HTTPException(status_code=503, detail="Voice service not configured")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    audio_bytes = await _voice_service.synthesize(req.text)

    log.info("voice.synthesized", bytes=len(audio_bytes))
    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(audio_bytes))},
    )
