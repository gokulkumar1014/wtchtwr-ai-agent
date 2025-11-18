import { useState } from "react";

interface ChatInputProps {
  disabled?: boolean;
  onSend: (text: string) => Promise<void> | void;
  error?: string | null;
}

export const ChatInput: React.FC<ChatInputProps> = ({ disabled, onSend, error }) => {
  const [value, setValue] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;
    const text = value.trim();
    setValue("");
    await onSend(text);
  };

  return (
    <div className="mt-4">
      {error && <div className="mb-2 text-sm text-red-500">{error}</div>}
      <form
        onSubmit={handleSubmit}
        className="sticky bottom-0 bg-white/95 backdrop-blur border border-primary-100 shadow-lg rounded-2xl flex items-center gap-3 px-5 py-3"
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask about occupancy, price comps, amenities, or reviews…"
          className="flex-1 bg-transparent outline-none text-base text-slate-700 placeholder:text-slate-400"
          disabled={disabled}
        />
        <button
          type="submit"
          disabled={disabled}
          className="px-5 py-2 rounded-full bg-primary-500 text-white font-semibold shadow-md hover:bg-primary-600 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
};
