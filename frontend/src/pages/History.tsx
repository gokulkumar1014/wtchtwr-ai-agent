import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteConversation, getConversation, listConversations, updateConversationTitle } from "@/lib/api";
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
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [savingId, setSavingId] = useState<string | null>(null);
  const [renameError, setRenameError] = useState<string | null>(null);

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
    const optimistic = conversations.filter((c) => c.id !== id);
    setConversations(optimistic);
    if (activeConversation?.id === id) {
      setActiveConversation(undefined as unknown as any);
    }
    try {
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
    } catch (error) {
      // if deletion fails, refetch list to stay consistent
      const data = await listConversations();
      setConversations(data);
    }
  };

  const beginRename = (conversationId: string, title: string) => {
    setEditingId(conversationId);
    setEditingValue(title);
    setRenameError(null);
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditingValue("");
    setRenameError(null);
  };

  const saveRename = async () => {
    if (!editingId) return;
    setSavingId(editingId);
    setRenameError(null);
    try {
      const updated = await updateConversationTitle(editingId, editingValue.trim());
      const updatedList = conversations.map((conversation) =>
        conversation.id === updated.id ? updated : conversation,
      );
      setConversations(updatedList);
      if (activeConversation?.id === updated.id) {
        setActiveConversation(updated);
      }
      setEditingId(null);
      setEditingValue("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to rename conversation.";
      setRenameError(message);
    } finally {
      setSavingId(null);
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
          const title = conversation.title?.trim() || preview;
          const isEditing = editingId === conversation.id;
          return (
            <div
              key={conversation.id}
              className="flex items-start gap-3 border border-slate-200 rounded-xl px-4 py-3 bg-white hover:border-primary-300 hover:shadow"
            >
              <button onClick={() => openConversation(conversation.id)} className="flex-1 text-left space-y-1">
                <div className="flex items-center gap-2">
                  {isEditing ? (
                    <>
                      <input
                        value={editingValue}
                        onChange={(event) => setEditingValue(event.target.value)}
                        maxLength={200}
                        className="flex-1 rounded-xl border border-slate-300 px-2 py-1 text-sm focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                        onClick={(event) => event.stopPropagation()}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            event.stopPropagation();
                            void saveRename();
                          }
                          if (event.key === "Escape") {
                            event.preventDefault();
                            event.stopPropagation();
                            cancelRename();
                          }
                        }}
                        autoFocus
                      />
                      <button
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          void saveRename();
                        }}
                        disabled={savingId === conversation.id}
                        className="rounded-full bg-primary-500 px-3 py-1 text-xs font-semibold text-white shadow hover:bg-primary-600 disabled:opacity-60"
                      >
                        {savingId === conversation.id ? "Saving…" : "Save"}
                      </button>
                      <button
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          cancelRename();
                        }}
                        className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-500 hover:border-slate-400"
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <div className="flex-1 text-sm font-semibold text-primary-600 truncate">{title}</div>
                  )}
                </div>
                <div className="text-xs text-slate-500 truncate">{preview}</div>
                <div className="text-[11px] text-slate-400">Last updated {formatTimestamp(conversation.updated_at)}</div>
                <div className="text-[11px] text-slate-400">{conversation.messages.length} total messages</div>
              </button>
              <div className="flex items-center gap-2">
                <button
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    beginRename(conversation.id, conversation.title ?? title);
                  }}
                  className="text-slate-400 hover:text-primary-600 text-base"
                  title="Rename chat"
                  aria-label="Rename chat"
                >
                  ✐
                </button>
                <button
                  onClick={() => removeConversation(conversation.id)}
                  className="text-slate-400 hover:text-red-500 text-base"
                  title="Delete conversation"
                >
                  🗑
                </button>
              </div>
            </div>
          );
        })}
        {!conversations.length && (
          <div className="text-slate-400 text-sm">No conversations yet.</div>
        )}
        {renameError && <div className="text-sm text-red-500">{renameError}</div>}
      </div>
    </div>
  );
};
