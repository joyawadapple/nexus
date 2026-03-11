import { useState } from "react";
import { useVoiceRecording } from "../hooks/useVoiceRecording";
import { transcribeAudio } from "../api/voice";

interface Props {
  onTranscript: (text: string) => void;
  disabled: boolean;
}

export default function VoiceButton({ onTranscript, disabled }: Props) {
  const { isRecording, startRecording, stopRecording, error } = useVoiceRecording();
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3500);
  };

  const handleClick = async () => {
    if (isTranscribing) return;

    if (!isRecording) {
      try {
        await startRecording();
      } catch {
        showToast(error || "Microphone access denied.");
      }
      return;
    }

    // Stop recording and transcribe
    try {
      const blob = await stopRecording();
      setIsTranscribing(true);
      const transcript = await transcribeAudio(blob);
      if (transcript) {
        onTranscript(transcript);
      } else {
        showToast("No speech detected. Please try again.");
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Transcription failed.");
    } finally {
      setIsTranscribing(false);
    }
  };

  const isActive = isRecording || isTranscribing;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled || isTranscribing}
        title={isRecording ? "Click to stop & send" : "Click to speak"}
        className={`
          flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all
          disabled:opacity-50 disabled:cursor-not-allowed
          ${isRecording
            ? "bg-red-600 hover:bg-red-700 text-white"
            : "bg-slate-700 hover:bg-slate-600 text-slate-200"
          }
        `}
      >
        {isTranscribing ? (
          <>
            <span className="w-4 h-4 border-2 border-slate-400 border-t-white rounded-full animate-spin" />
          </>
        ) : isRecording ? (
          <>
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span className="text-xs">Recording</span>
          </>
        ) : (
          <MicIcon />
        )}
      </button>

      {isActive && !isTranscribing && (
        <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
        </span>
      )}

      {toastMsg && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-max max-w-xs bg-slate-700 text-slate-200 text-xs px-3 py-1.5 rounded-lg shadow-lg border border-slate-600 z-50">
          {toastMsg}
        </div>
      )}
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="w-4 h-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M19 10a7 7 0 0 1-14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}
