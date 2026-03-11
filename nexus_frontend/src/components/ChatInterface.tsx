import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentStatus } from "../api/nexus";
import VoiceButton from "./VoiceButton";
import { synthesizeText } from "../api/voice";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface Props {
  sessionId: string | null;
  company: string;
  tier: string;
  onSendMessage: (msg: string) => Promise<void>;
  messages: Message[];
  isLoading: boolean;
  status: string;
  agentStatuses: AgentStatus[];
}

const TIER_COLORS: Record<string, string> = {
  platinum: "bg-purple-600",
  gold: "bg-yellow-600",
  standard: "bg-slate-600",
};

const STATUS_COLORS: Record<string, string> = {
  active: "text-blue-400",
  collecting: "text-yellow-400",
  in_progress: "text-blue-400",
  complete: "text-green-400",
  escalated: "text-red-400",
  known_incident: "text-orange-400",
};

export default function ChatInterface({
  sessionId,
  company,
  tier,
  onSendMessage,
  messages,
  isLoading,
  status,
  agentStatuses: _agentStatuses,
}: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastInputWasVoice = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-play TTS when the latest assistant message arrived via a voice input
  useEffect(() => {
    if (!lastInputWasVoice.current) return;
    const last = messages[messages.length - 1];
    if (!last || last.role !== "assistant") return;
    synthesizeText(last.content)
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.play().catch(() => {});
      })
      .catch(() => {});
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    lastInputWasVoice.current = false;
    const msg = input;
    setInput("");
    await onSendMessage(msg);
  };

  const handleVoiceTranscript = async (transcript: string) => {
    lastInputWasVoice.current = true;
    await onSendMessage(transcript);
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold">
            N
          </div>
          <div>
            <div className="font-semibold text-sm">Nexus Support</div>
            <div className="text-xs text-slate-400">{company || "Connecting..."}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {tier && (
            <span className={`text-xs px-2 py-1 rounded-full text-white ${TIER_COLORS[tier] || "bg-slate-600"}`}>
              {tier.toUpperCase()}
            </span>
          )}
          {status && (
            <span className={`text-xs font-medium ${STATUS_COLORS[status] || "text-slate-400"}`}>
              {status.replace("_", " ").toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-8 text-sm">
            Start your support session by describing your issue.
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-100 border border-slate-700"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="text-xs text-blue-400 font-medium mb-1">NEXUS</div>
              )}
              {msg.role === "assistant" ? (
                <div className="prose prose-sm prose-invert max-w-none
                  prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1
                  prose-li:my-0 prose-code:text-blue-300 prose-code:bg-slate-700
                  prose-code:px-1 prose-code:rounded prose-pre:bg-slate-700
                  prose-pre:border prose-pre:border-slate-600
                  prose-a:text-blue-400 prose-strong:text-slate-100
                  prose-blockquote:border-blue-500 prose-blockquote:text-slate-300">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              )}
              <div className="text-xs opacity-50 mt-1 text-right">
                {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-3">
              <div className="text-xs text-blue-400 font-medium mb-1">NEXUS</div>
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-700">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={sessionId ? "Describe your issue..." : "Start a session first"}
            disabled={!sessionId || isLoading}
            className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
          <VoiceButton
            onTranscript={handleVoiceTranscript}
            disabled={!sessionId || isLoading}
          />
          <button
            type="submit"
            disabled={!sessionId || isLoading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
