import { useEffect, useMemo, useState } from "react";
import { Route, Routes, useNavigate } from "react-router-dom";
import { ChatPage } from "@/pages/Chat";
import { DashboardPage } from "@/pages/Dashboard";
import { HistoryPage } from "@/pages/History";
import { DataExportPage } from "@/pages/DataExport";
import {
  createConversation,
  deleteMessage as apiDeleteMessage,
  getConversation,
  listConversations,
} from "@/lib/api";
import { Conversation, Message, useChatStore } from "@/store/useChat";
import { useUiStore } from "@/store/useUi";
import { ABOUT_POINTS, HELP_INTRO, SAMPLE_QUERIES } from "@/content";
import { API_BASE_URL } from "@/lib/api";

const SIDEBAR_WIDTH = 300;
const BUTTON_BASE =
  "w-full block text-center rounded-xl px-4 py-3 border border-primary-100 bg-white/90 text-slate-600 font-semibold transition hover:border-primary-400 hover:text-primary-600 hover:shadow";

export default function App(): JSX.Element {
  const navigate = useNavigate();
  const {
    setConversations,
    setActiveConversation,
    conversations,
    activeConversation,
    setFocusMessageId,
  } = useChatStore();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { helpOpen, setHelpOpen } = useUiStore();
  const [showAbout, setShowAbout] = useState(false);
  const lastMessageId = (conversation?: Conversation) =>
    conversation && conversation.messages.length
      ? conversation.messages[conversation.messages.length - 1]?.id
      : undefined;

  const openSlackbot = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/slackbot/start`, { method: "POST" });
    } catch (error) {
      console.warn("Unable to start Slackbot", error);
    }
    window.open("https://app.slack.com/client/T09M3FZUMQ8/D09TXFD3S0K", "_blank", "noopener");
    window.setTimeout(() => {
      window.open("slack://channel?team=T09M3FZUMQ8&id=D09TXFD3S0K", "_blank");
    }, 150);
  };

  useEffect(() => {
    (async () => {
      try {
        const data = await listConversations();
        if (data.length) {
          setConversations(data);
          setActiveConversation(data[0]);
          setFocusMessageId(lastMessageId(data[0]));
          return;
        }
      } catch (error) {
        console.error("Unable to load conversations", error);
      }
      try {
        const created = await createConversation();
        const full = await getConversation(created.id);
        setConversations([full]);
        setActiveConversation(full);
        setFocusMessageId(lastMessageId(full));
      } catch (error) {
        console.error("Unable to initialise conversation", error);
      }
    })();
  }, [setActiveConversation, setConversations, setFocusMessageId]);

  const createNewChat = async () => {
    const created = await createConversation();
    const full = await getConversation(created.id);
    const updated = [full, ...conversations.filter((c) => c.id !== full.id)];
    setConversations(updated);
    setActiveConversation(full);
    setFocusMessageId(lastMessageId(full));
    setSidebarCollapsed(false);
    navigate("/");
  };

  const openConversation = async (id: string, focusId?: string) => {
    const convo = await getConversation(id);
    const others = conversations.filter((c) => c.id !== id);
    setConversations([convo, ...others]);
    setActiveConversation(convo);
    setFocusMessageId(focusId ?? lastMessageId(convo));
    setSidebarCollapsed(false);
    navigate("/");
  };

  const recentMessages = useMemo(() => {
    const collected: { message: Message; conversation: Conversation }[] = [];
    conversations.forEach((conversation) => {
      conversation.messages
        .filter((message) => message.role === "user" && message.content)
        .forEach((message) => collected.push({ message, conversation }));
    });
    return collected
      .sort((a, b) => new Date(b.message.timestamp).getTime() - new Date(a.message.timestamp).getTime())
      .slice(0, 10);
  }, [conversations]);

  const handleDeleteMessage = async (conversationId: string, messageId: string) => {
    // Optimistic removal for snappier UI
    const optimistic = conversations.map((c) => {
      if (c.id !== conversationId) return c;
      return {
        ...c,
        messages: c.messages.filter((m) => m.id !== messageId),
      };
    });
    setConversations(optimistic);
    setFocusMessageId(undefined);
    try {
      const updated = await apiDeleteMessage(conversationId, messageId);
      const others = conversations.filter((c) => c.id !== updated.id);
      const merged = [updated, ...others];
      setConversations(merged);
      if (activeConversation?.id === conversationId) {
        setActiveConversation(updated);
      }
    } catch (error) {
      // On failure, refetch the conversation to stay consistent
      const convo = await getConversation(conversationId);
      const others = conversations.filter((c) => c.id !== convo.id);
      setConversations([convo, ...others]);
      if (activeConversation?.id === conversationId) {
        setActiveConversation(convo);
      }
    }
  };

  const formatTimestamp = (iso: string) =>
    new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York" });

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-slate-50 via-indigo-50 to-white">
      <div className="h-full flex gap-6 px-6 py-6">
        {!sidebarCollapsed ? (
          <aside
            className="relative flex flex-col gap-6 bg-white/90 border border-primary-100 shadow-2xl rounded-3xl px-6 pt-6 pb-6 overflow-hidden backdrop-blur max-h-full"
            style={{ width: SIDEBAR_WIDTH, minWidth: 260 }}
          >
            <div className="overflow-y-auto pr-2 flex-1 flex flex-col gap-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <img
                  src="/specula-logo.png"
                  alt="wtchtwr"
                  className="w-20 h-20 rounded-3xl object-contain border border-primary-100 bg-white shadow"
                />
                <img
                  src="/logo.svg"
                  alt="Highbury"
                  className="w-20 h-20 rounded-3xl object-contain border border-slate-200 bg-white shadow"
                />
              </div>
              <button
                onClick={() => setSidebarCollapsed(true)}
                className="text-slate-400 hover:text-slate-600 text-xl"
                title="Collapse panel"
              >
                ×
              </button>
            </div>
            <div className="space-y-1 text-center">
              <div className="text-base font-semibold tracking-[0.22em] text-primary-500">wtchtwr</div>
              <p className="text-xs text-slate-500">Highbury’s demo portfolio co-pilot for NYC</p>
            </div>

            <div className="space-y-3">
              <button
                onClick={createNewChat}
                className="w-full rounded-xl px-4 py-3 bg-primary-500 text-white font-semibold shadow-md hover:bg-primary-600"
              >
                ＋ New Chat
              </button>
              <button onClick={() => navigate("/history")} className={BUTTON_BASE}>
                Chat History
              </button>
              <button onClick={() => navigate("/dashboard")} className={BUTTON_BASE}>
                Dashboard
              </button>
              <button onClick={() => navigate("/data-export")} className={BUTTON_BASE}>
                Data Export
              </button>
              <button onClick={openSlackbot} className={BUTTON_BASE}>
                Open Slackbot
              </button>
              <button onClick={() => setHelpOpen(true)} className={BUTTON_BASE}>
                Help &amp; Ideas
              </button>
            </div>

            <div className="mt-auto border border-slate-200 rounded-xl bg-white/95 p-4">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Recent messages</div>
              <div className="mt-3 space-y-2 max-h-64 overflow-y-auto pr-1">
                {recentMessages.length === 0 && (
                  <div className="text-xs text-slate-400">Send a question to populate this list.</div>
                )}
                {recentMessages.map(({ message, conversation }) => (
                  <div
                    key={message.id}
                    className="flex items-start gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs hover:border-primary-300"
                  >
                    <button
                      onClick={() => openConversation(conversation.id, message.id)}
                      className="flex-1 text-left space-y-1"
                    >
                      <div className="font-semibold text-slate-600 text-sm break-words">
                        {message.content?.replace(/\s+/g, " ") || conversation.title}
                      </div>
                      <div className="text-[10px] text-slate-400">{formatTimestamp(message.timestamp)}</div>
                    </button>
                    <button
                      onClick={() => handleDeleteMessage(conversation.id, message.id)}
                      className="text-slate-400 hover:text-red-500"
                      title="Delete message"
                    >
                      🗑
                    </button>
                  </div>
                ))}
              </div>
            </div>
            </div>
          </aside>
        ) : (
          <button
            onClick={() => setSidebarCollapsed(false)}
            className="fixed top-6 left-6 z-20 flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-4 py-2 shadow text-slate-600 hover:text-slate-800"
            title="Open menu"
          >
            ☰ Menu
          </button>
        )}

        <main
          className={`flex-1 bg-white/95 border border-primary-100 shadow-2xl rounded-3xl px-8 py-6 transition-all flex flex-col overflow-hidden ${
            sidebarCollapsed ? "ml-28" : ""
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-200 mb-6">
            <div>
              <div className="text-3xl font-bold text-slate-800 tracking-[0.22em]">wtchtwr</div>
            </div>
            <button
              onClick={() => setShowAbout(true)}
              className="rounded-full border border-primary-200 px-4 py-2 text-sm font-semibold text-primary-600 hover:bg-primary-50"
            >
              About wtchtwr
            </button>
          </div>
          <div className="flex-1 overflow-hidden min-h-0">
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/data-export" element={<DataExportPage />} />
            </Routes>
          </div>
        </main>
      </div>
      {helpOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/50 px-6">
          <div className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-3xl bg-white border border-primary-100 shadow-2xl p-8 space-y-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-2xl font-semibold text-slate-800">Jump-start your prompts</h3>
                <p className="text-sm text-slate-500 mt-1">{HELP_INTRO}</p>
              </div>
              <button
                onClick={() => setHelpOpen(false)}
                className="text-slate-400 hover:text-slate-600 text-xl"
                aria-label="Close help"
              >
                ✕
              </button>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 shadow-inner">
              <p className="text-sm font-semibold text-slate-700 mb-2">Watch the wtchtwr walkthrough</p>
              <video src="/demo-video.mp4" controls className="w-full rounded-xl shadow max-h-64">
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[60vh] overflow-y-auto pr-1">
              {SAMPLE_QUERIES.map((entry, idx) => (
                <div
                  key={`${entry}-${idx}`}
                  className="rounded-2xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-700 shadow-sm"
                >
                  <span className="text-xs font-semibold text-primary-500 mr-2">#{idx + 1}</span>
                  {entry}
                </div>
              ))}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setHelpOpen(false)}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      {showAbout && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/50 px-6">
          <div className="w-full max-w-3xl rounded-3xl bg-white border border-primary-100 shadow-2xl p-8 space-y-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-2xl font-semibold text-slate-800">About wtchtwr</h3>
                <p className="text-sm text-slate-500 mt-1">
                  wtchtwr is our full-stack portfolio assistant. We prototyped it for Highbury - our fictional
                  hospitality client - to demonstrate how operators can interrogate NYC listings data in real time.
                </p>
            </div>
              <button
                onClick={() => setShowAbout(false)}
                className="text-slate-400 hover:text-slate-600 text-xl"
                aria-label="Close about"
              >
                ✕
              </button>
            </div>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-1">
              {ABOUT_POINTS.map((section) => (
                <div
                  key={section.title}
                  className="rounded-2xl border border-slate-200 bg-slate-50/70 px-5 py-4 shadow-sm"
                >
                  <h4 className="text-base font-semibold text-slate-800">{section.title}</h4>
                  <p className="mt-1 text-sm text-slate-600 leading-relaxed">{section.detail}</p>
                </div>
              ))}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setShowAbout(false)}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
