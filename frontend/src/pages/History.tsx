import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { deleteConversation, getConversation, listConversations } from "@/lib/api";
import { useChatStore } from "@/store/useChat";

const formatTimestamp = (iso: string) =>
  new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York" });

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    conversations,
    setConversations,
    setActiveConversation,
    setFocusMessageId,
    activeConversation,
  } = useChatStore();

  useEffect(() => {
    (async () => {
      const data = await listConversations();
      setConversations(data);
    })();
  }, [setConversations]);

  const openConversation = async (id: string) => {
    const conversation = await getConversation(id);
    setActiveConversation(conversation);
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    setFocusMessageId(lastMessage?.id);
    navigate("/");
  };

  const removeConversation = async (id: string) => {
    const remaining = await deleteConversation(id);
    setConversations(remaining);
    const hasAny = remaining.length > 0;
    const hasActive = remaining.some((conversation) => conversation.id === activeConversation?.id);
    if (hasAny && !hasActive) {
      setActiveConversation(remaining[0]);
      const fallbackLast = remaining[0].messages[remaining[0].messages.length - 1];
      setFocusMessageId(fallbackLast?.id);
    }
    const hasUserChats = remaining.some((conversation) =>
      conversation.messages.some((message) => message.role === "user" && message.content),
    );
    if (!hasAny || !hasUserChats) {
      const fallback = remaining[0];
      if (fallback) {
        setActiveConversation(fallback);
        const fallbackLast = fallback.messages[fallback.messages.length - 1];
        setFocusMessageId(fallbackLast?.id);
      }
      navigate("/");
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-primary-600">Conversation history</h1>
          <p className="text-slate-500 text-sm">
            Chats are retained for 30 days. Select one to reopen it or remove it from the archive.
          </p>
        </div>
        <button
          onClick={() => navigate(-1)}
          className="inline-flex items-center gap-2 rounded-full border border-primary-200 px-4 py-2 text-sm font-semibold text-primary-600 hover:bg-primary-50"
        >
          ← Back
        </button>
      </div>
      <div className="flex-1 overflow-y-auto pr-2 space-y-3">
        {conversations.map((conversation) => {
          const firstUserMessage = conversation.messages.find(
            (message) => message.role === "user" && message.content,
          );
          const preview = firstUserMessage?.content?.replace(/\s+/g, " ") ?? "No user messages yet.";
          return (
            <div
              key={conversation.id}
              className="flex items-start gap-3 border border-slate-200 rounded-xl px-4 py-3 bg-white hover:border-primary-300 hover:shadow"
            >
              <button onClick={() => openConversation(conversation.id)} className="flex-1 text-left space-y-1">
                <div className="text-sm font-semibold text-primary-600 truncate">{preview}</div>
                <div className="text-[11px] text-slate-400">Last updated {formatTimestamp(conversation.updated_at)}</div>
                <div className="text-[11px] text-slate-400">{conversation.messages.length} total messages</div>
              </button>
              <button
                onClick={() => removeConversation(conversation.id)}
                className="text-slate-400 hover:text-red-500"
                title="Delete conversation"
              >
                🗑
              </button>
            </div>
          );
        })}
        {!conversations.length && (
          <div className="text-slate-400 text-sm">No conversations yet.</div>
        )}
      </div>
    </div>
  );
};
