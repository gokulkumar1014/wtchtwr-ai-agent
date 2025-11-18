import type { Conversation, Message } from "@/store/useChat";

const DEBUG = import.meta.env.VITE_DEBUG === "true";

export const hydrateMessage = (raw: Message): Message => {
  if (!raw) return raw;
  const message: Message = { ...raw };
  if (typeof message.payload === "string") {
    try {
      const parsed = JSON.parse(message.payload);
      message.payload = parsed && typeof parsed === "object" ? parsed : {};
      if (DEBUG && message.role === "assistant") {
        console.info(
          "[HYDRATE_FIX][PARSE]",
          message.role,
          "decoded payload keys:",
          Object.keys(message.payload as Record<string, unknown>),
        );
      }
    } catch (error) {
      console.warn("[HYDRATE_FIX][PARSE_ERROR]", error);
      message.payload = {};
    }
  }
  if (!message.payload || typeof message.payload !== "object") {
    message.payload = {};
  }
  return message;
};

export const normalizeConversation = (conversation: Conversation): Conversation => ({
  ...conversation,
  messages: Array.isArray(conversation.messages)
    ? conversation.messages.map((msg) => hydrateMessage(msg))
    : [],
});
