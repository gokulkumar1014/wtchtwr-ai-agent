import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useChatStore } from "@/store/useChat";
import { MessageBubble } from "@/components/Message";
import { ChatInput } from "@/components/ChatInput";
import {
  API_BASE_URL,
  exportMessage,
  sendSummaryEmail,
  summarizeConversation,
  updateConversationTitle,
  type ConversationSummaryPayload,
} from "@/lib/api";
import { WELCOME_MESSAGE } from "@/content";
import { useUiStore } from "@/store/useUi";
import type { AssistantPayload, Conversation, Message } from "@/store/useChat";
import { hydrateMessage } from "@/lib/messages";
const DEBUG = import.meta.env.VITE_DEBUG === "true";

const SLACKBOT_URL = "https://app.slack.com/client/T09M3FZUMQ8/D09TXFD3S0K";
const SLACKBOT_APP_LINK = "slack://channel?team=T09M3FZUMQ8&id=D09TXFD3S0K";
const THREAD_STORAGE_KEY = "wtchtwr.chat.thread_id";
const THINKING_TRACE_ENABLED = (import.meta.env.VITE_THINKING_TRACE || "on") !== "off";

type StructuredSummaryItem = {
  text: string;
  kind: "heading" | "bullet";
  depth: number;
};

type SummaryListOptions = {
  allowHeadings?: boolean;
};

const SUMMARY_HEADING_KEYWORDS = [
  "key insights",
  "overview",
  "summary",
  "focus area",
  "focus areas",
  "focus",
  "areas for improvement",
  "improvement needed",
  "positive feedback",
  "concerns noted",
  "recommendations",
  "next steps",
  "strategy",
  "strategies",
  "action plan",
  "evaluation",
  "insights",
];

const SUMMARY_HEADING_PREFIXES = SUMMARY_HEADING_KEYWORDS.map((keyword) => keyword.toLowerCase());

const sanitizeHeading = (text: string) => text.replace(/[:：]\s*$/, "").trim();

const toStructuredSummaryItem = (raw: string, allowHeadings: boolean): StructuredSummaryItem | null => {
  const trimmed = (raw || "").trim();
  if (!trimmed) return null;

  const lower = trimmed.toLowerCase();
  const numericMatch = trimmed.match(/^(\d+(?:\.\d+)*)(?:[\.\)]\s*)?/);
  const numericDepth = numericMatch ? Math.min(numericMatch[1].split(".").length, 2) : 0;

  const looksLikeKeywordHeading =
    allowHeadings &&
    SUMMARY_HEADING_PREFIXES.some(
      (keyword) => lower === keyword || lower.startsWith(`${keyword}:`),
    );
  const endsWithColon = allowHeadings && /[:：]\s*$/.test(trimmed);
  const numberedHeading = allowHeadings && /^\d+(\.\d+)*[\.\)]/.test(trimmed);

  if (looksLikeKeywordHeading || endsWithColon || numberedHeading) {
    return {
      text: sanitizeHeading(trimmed),
      kind: "heading",
      depth: numericDepth,
    };
  }

  return {
    text: trimmed,
    kind: "bullet",
    depth: numericDepth,
  };
};

const buildStructuredSummaryItems = (items?: string[], allowHeadings = true): StructuredSummaryItem[] => {
  if (!items || items.length === 0) return [];
  return items
    .map((item) => toStructuredSummaryItem(item, allowHeadings))
    .filter((item): item is StructuredSummaryItem => Boolean(item && item.text));
};

type ShareOptionValue = "sql" | "csv" | "both";

interface EmailModalState {
  mode: "summary" | "result";
  email: string;
  variant?: "concise" | "detailed";
  messageId?: string;
  tableIndex?: number;
  label?: string;
  shareOption?: ShareOptionValue;
  allowedShareOptions?: ShareOptionValue[];
}

const sanitizeMessages = (messages: Message[]): Message[] => {
  if (!Array.isArray(messages)) {
    return [];
  }
  const cleaned: Message[] = [];
  let index = 0;
  while (index < messages.length) {
    const current = messages[index];
    if (
      current.role === "user" &&
      typeof current.content === "string" &&
      current.content.trim() &&
      index + 3 < messages.length
    ) {
      const errorAssistant = messages[index + 1];
      const retryUser = messages[index + 2];
      const retryAssistant = messages[index + 3];
      const retryType = retryAssistant?.payload?.response_type;
      if (
        errorAssistant?.role === "assistant" &&
        errorAssistant.payload?.response_type === "error" &&
        typeof retryUser?.content === "string" &&
        retryUser.content.trim() === current.content.trim() &&
        retryAssistant?.role === "assistant" &&
        retryType !== "error"
      ) {
        index += 2;
        continue;
      }
    }
    cleaned.push(current);
    index += 1;
  }
  return cleaned;
};

const prepareConversation = (conversation: Conversation): Conversation => ({
  ...conversation,
  messages: sanitizeMessages(conversation.messages).map(hydrateMessage),
});

export const ChatPage: React.FC = () => {
  const {
    activeConversation,
    setActiveConversation,
    conversations,
    setConversations,
    focusMessageId,
    setFocusMessageId,
  } = useChatStore();
  const navigate = useNavigate();
  const { setHelpOpen } = useUiStore();
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(true);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<string[]>([]);

  const [summaryData, setSummaryData] = useState<ConversationSummaryPayload | null>(null);
  const [summaryView, setSummaryView] = useState<"concise" | "detailed">("concise");
  const [summaryPending, setSummaryPending] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [exportLoading, setExportLoading] = useState<{ messageId: string; tableIndex: number } | null>(null);
  const [exportError, setExportError] = useState<{ messageId: string; tableIndex: number; message: string } | null>(
    null,
  );

  const [emailModal, setEmailModal] = useState<EmailModalState | null>(null);
  const [emailSending, setEmailSending] = useState(false);
  const [emailFeedback, setEmailFeedback] = useState<string | null>(null);
  const [emailError, setEmailError] = useState<string | null>(null);
  const [titleEditing, setTitleEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [titleSaving, setTitleSaving] = useState(false);
  const [titleError, setTitleError] = useState<string | null>(null);
  const actionHandledRef = useRef<Set<string>>(new Set());
  const EMAIL_SHARE_OPTIONS: { value: ShareOptionValue; label: string; description: string }[] = [
    { value: "sql", label: "SQL only", description: "Embed the generated SQL in the email body." },
    { value: "csv", label: "CSV only", description: "Attach the selected table as a CSV file." },
    { value: "both", label: "SQL + CSV", description: "Send the SQL inline and attach the CSV export." },
  ];
  const highlightCleanupRef = useRef<(() => void) | null>(null);
  const highlightFadeTimeoutRef = useRef<number | null>(null);
  const autoScrollLockRef = useRef(false);
  const scrollRetryRef = useRef<number | null>(null);
  const pendingMessageRef = useRef<{
    conversationId: string;
    userId: string;
    assistantId: string;
    prompt: string;
    streamContent: string;
  } | null>(null);
  const [lastResponse, setLastResponse] = useState<{ policy?: string; telemetry?: Record<string, unknown> } | null>(
    null,
  );
  const [threadId] = useState(() => {
    if (typeof window === "undefined") {
      return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    const existing = window.localStorage.getItem(THREAD_STORAGE_KEY);
    if (existing) return existing;
    const generated =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.localStorage.setItem(THREAD_STORAGE_KEY, generated);
    return generated;
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THREAD_STORAGE_KEY, threadId);
    }
  }, [threadId]);

  const clearHighlight = useCallback(
    (unlock: boolean) => {
      if (highlightFadeTimeoutRef.current) {
        window.clearTimeout(highlightFadeTimeoutRef.current);
        highlightFadeTimeoutRef.current = null;
      }
      setHighlightedIds([]);
      if (highlightCleanupRef.current) {
        highlightCleanupRef.current();
        highlightCleanupRef.current = null;
      }
      if (unlock) {
        autoScrollLockRef.current = false;
      }
    },
    []
  );

  const currentConversation = useMemo(() => {
    if (!activeConversation) return undefined;
    const latest =
      conversations.find((conversation) => conversation.id === activeConversation.id) ?? activeConversation;
    return prepareConversation(latest);
  }, [activeConversation, conversations]);

  useEffect(() => {
    if (!currentConversation) return;
    console.warn(
      "[HYDRATE_DEBUG] Loaded conversation",
      currentConversation.id,
      "with messages:",
      currentConversation.messages?.length ?? 0,
    );
    currentConversation.messages?.forEach((msg, i) => {
      console.warn(`[HYDRATE_DEBUG] Message[${i}]`, {
        role: msg.role,
        hasContent: Boolean(msg.content),
        hasPayload: Boolean(msg.payload),
        payloadKeys: msg.payload ? Object.keys(msg.payload) : [],
      });
    });
  }, [currentConversation]);

  const generateTempId = () => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };

  const injectLocalErrorMessage = (conversation: Conversation, question: string, detail: string) => {
    const now = new Date().toISOString();
    const userMessageId = generateTempId();
    const assistantMessageId = generateTempId();
    const localConversation: Conversation = {
      ...conversation,
      messages: [
        ...conversation.messages,
        {
          id: userMessageId,
          role: "user",
          content: question,
          timestamp: now,
        },
        {
          id: assistantMessageId,
          role: "assistant",
          nl_summary: detail,
          payload: {
            tables: [],
            summary: detail,
            question,
            response_type: "error",
          },
          timestamp: now,
        },
      ],
      updated_at: now,
    };
    syncConversation(localConversation);
  };

  const settlePendingAsError = (question: string, detail: string): boolean => {
    const pending = pendingMessageRef.current;
    if (!currentConversation || !pending || currentConversation.id !== pending.conversationId) {
      return false;
    }
    const { assistantId, prompt } = pending;
    const now = new Date().toISOString();
    const updatedMessages = currentConversation.messages.map((message) => {
      if (message.id !== assistantId) return message;
      return {
        ...message,
        nl_summary: detail,
        payload: {
          tables: [],
          summary: detail,
          question: question || prompt,
          response_type: "error",
        },
        timestamp: now,
      };
    });
    const updatedConversation: Conversation = {
      ...currentConversation,
      messages: updatedMessages,
      updated_at: now,
    };
    syncConversation(updatedConversation);
    pendingMessageRef.current = null;
    return true;
  };

  useEffect(() => {
    if (!activeConversation && conversations.length) {
      setActiveConversation(prepareConversation(conversations[0]));
    }
  }, [activeConversation, conversations, setActiveConversation]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!currentConversation) {
      actionHandledRef.current.clear();
      return;
    }
    const handled = new Set<string>();
    currentConversation.messages.forEach((message) => {
      if (message.role === "assistant" && message.payload?.action) {
        handled.add(message.id);
      }
    });
    actionHandledRef.current = handled;
  }, [currentConversation?.id]);

  useEffect(() => {
    if (!currentConversation) {
      setTitleEditing(false);
      setTitleDraft("");
      setTitleError(null);
      return;
    }
    if (!titleEditing) {
      setTitleDraft(currentConversation.title || "");
    }
  }, [currentConversation?.id, currentConversation?.title, titleEditing]);

  const userMessageCount = useMemo(
    () =>
      currentConversation?.messages.reduce(
        (count, message) => (message.role === "user" && message.content ? count + 1 : count),
        0,
      ) ?? 0,
    [currentConversation],
  );

  useEffect(() => {
    if (focusMessageId) return;
    const container = messagesContainerRef.current;
    if (!container || autoScrollLockRef.current) return;
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [currentConversation?.messages, focusMessageId]);

  const syncConversation = (conversation: Conversation) => {
    const sanitized = prepareConversation(conversation);
    setActiveConversation(sanitized);
    const { conversations: existing } = useChatStore.getState();
    const others = existing.filter((c) => c.id !== conversation.id).map(prepareConversation);
    setConversations([sanitized, ...others]);
    if (DEBUG) {
      console.debug("[STATE_UPDATE]", {
        messages: sanitized.messages.length,
        conversationId: sanitized.id,
      });
    }
  };

  const handleCommand = (raw: string): "dashboard" | "slackbot" | "help" | null => {
    const normalized = raw.trim().toLowerCase();
    const command = normalized === "dashbard" ? "dashboard" : normalized;
    if (command === "dashboard") {
      navigate("/dashboard");
      return "dashboard";
    }
    if (command === "help") {
      setHelpOpen(true);
      return "help";
    }
    if (command === "slackbot") {
      fetch(`${API_BASE_URL}/api/slackbot/start`, { method: "POST" }).catch(() => undefined);
      window.open(SLACKBOT_URL, "_blank", "noopener");
      window.setTimeout(() => {
        window.open(SLACKBOT_APP_LINK, "_blank");
      }, 150);
      return "slackbot";
    }
    return null;
  };

  const handleSend = async (text: string) => {
    if (!currentConversation) return;
    const trimmed = text.trim();
    if (!trimmed) return;
    const command = handleCommand(trimmed);
    const isCommand = Boolean(command);
    const conversationId = currentConversation.id;
    if (isMountedRef.current) {
      setLoading(true);
      setErrorMessage(null);
    }
    // Optimistic UI: append the user's message immediately + a fetching placeholder
    const optimisticUserId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-user`;
    const optimisticAssistantId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-assistant`;
    if (!isCommand && currentConversation) {
      const now = new Date().toISOString();
      const optimistic = {
        ...currentConversation,
        messages: [
          ...currentConversation.messages,
          { id: optimisticUserId, role: "user", content: trimmed, timestamp: now },
          {
            id: optimisticAssistantId,
            role: "assistant",
            nl_summary: "Fetching result…",
            payload: { tables: [], response_type: "loading", summary: "" },
            timestamp: now,
          },
        ],
        updated_at: now,
      } as Conversation;
      syncConversation(optimistic);
      pendingMessageRef.current = {
        conversationId,
        userId: optimisticUserId,
        assistantId: optimisticAssistantId,
        prompt: trimmed,
        streamContent: "",
      };
      // keep the list scrolled to bottom while fetching
      const container = messagesContainerRef.current;
      if (container) {
        container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
      }
    }
    if (isCommand) {
      if (isMountedRef.current) {
        setLoading(false);
      }
      setLastResponse(null);
      return;
    }
    const startedAt = typeof performance !== "undefined" ? performance.now() : Date.now();

    const parseJsonSafe = (raw: string) => {
      try {
        return JSON.parse(raw);
      } catch {
        return raw;
      }
    };

    const appendAssistantToken = (token: string) => {
      if (!token) return;
      const pending = pendingMessageRef.current;
      if (!pending || pending.conversationId !== conversationId) {
        return;
      }
      const state = useChatStore.getState();
      const active =
        state.activeConversation && state.activeConversation.id === pending.conversationId
          ? state.activeConversation
          : undefined;
      const sourceConversation =
        active ?? state.conversations.find((conv) => conv.id === pending.conversationId);
      if (!sourceConversation) {
        return;
      }
      const now = new Date().toISOString();
      const updatedMessages = sourceConversation.messages.map((message) => {
        if (message.id !== pending.assistantId) return message;
        const previous = message.content ?? "";
        const nextContent = `${previous}${token}`;
        const payload: AssistantPayload & { streaming?: boolean } = {
          ...(message.payload ?? {}),
          summary: nextContent,
          question: pending.prompt,
          response_type: "streaming",
        };
        payload.streaming = true;
        pending.streamContent = nextContent;
        return {
          ...message,
          content: nextContent,
          nl_summary: nextContent,
          payload,
        };
      });
      const updatedConversation: Conversation = {
        ...sourceConversation,
        messages: updatedMessages,
        updated_at: now,
      };
      syncConversation(updatedConversation);
    };

    const handleFinalConversation = (value: unknown) => {
      if (!value || typeof value !== "object") {
        return;
      }

      const pending = pendingMessageRef.current;
      const prepared = prepareConversation(value as Conversation);
      if (pending && pending.streamContent) {
        const latestAssistant = [...prepared.messages].filter((msg) => msg.role === "assistant").pop();
        if (latestAssistant) {
          latestAssistant.content = pending.streamContent;
          latestAssistant.nl_summary = pending.streamContent;
        }
      }
      syncConversation(prepared);
      const latestAssistant = [...prepared.messages].filter((msg) => msg.role === "assistant").pop();
      const latestPayload = latestAssistant?.payload as AssistantPayload | undefined;
      const policyLabel =
        typeof latestPayload?.policy === "string"
          ? latestPayload.policy
          : typeof (latestPayload as Record<string, unknown> | undefined)?.policy === "string"
            ? ((latestPayload as Record<string, unknown>).policy as string)
            : undefined;
      setLastResponse({ policy: policyLabel, telemetry: latestPayload?.telemetry });
      pendingMessageRef.current = null;
    };

    const streamResponse = async () => {
      const response = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: trimmed,
          stream: true,
          thread_id: threadId,
          debug_thinking: THINKING_TRACE_ENABLED,
        }),
      });
      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const errorText = await response.text();
          if (errorText) {
            const parsed = parseJsonSafe(errorText);
            if (parsed && typeof parsed === "object") {
              detail =
                (parsed as Record<string, unknown>).detail?.toString() ??
                (parsed as Record<string, unknown>).message?.toString() ??
                (parsed as Record<string, unknown>).error?.toString() ??
                detail;
            } else if (typeof parsed === "string" && parsed.trim()) {
              detail = parsed;
            }
          }
        } catch {
          // ignore parse failures and fall back to default detail
        }
        throw new Error(detail);
      }

      const body = response.body;
      if (!body) {
        throw new Error("Streaming response missing body.");
      }

      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffered = "";
      let finalReceived = false;
      let doneReceived = false;

      const handleParsedData = (parsed: unknown) => {
        if (parsed === null || parsed === undefined) {
          return;
        }
        if (typeof parsed === "string") {
          if (parsed.trim()) {
            appendAssistantToken(parsed);
          }
          return;
        }
        if (typeof parsed !== "object") {
          return;
        }
        const record = parsed as Record<string, unknown>;
        const type = typeof record.type === "string" ? (record.type as string) : undefined;
        if (DEBUG) {
          console.info(`[STREAM_EVENT] ${type ?? "unknown"}`, {
            hasPayload: Boolean(record.payload),
            keys: record.payload && typeof record.payload === "object" ? Object.keys(record.payload) : undefined,
          });
        }
        if (type === "token") {
          const token =
            typeof record.content === "string"
              ? (record.content as string)
              : typeof record.payload === "string"
                ? (record.payload as string)
                : typeof record.token === "string"
                  ? (record.token as string)
                  : "";
          if (token) {
            appendAssistantToken(token);
          }
          return;
        }
        if (type === "final") {
          finalReceived = true;
          setErrorMessage(null);
          handleFinalConversation(record.payload);
          return;
        }
        if (type === "error") {
          const detail =
            typeof record.error === "string"
              ? (record.error as string)
              : typeof record.message === "string"
                ? (record.message as string)
                : "Streaming error.";
          setErrorMessage(detail);
          if (record.payload) {
            handleFinalConversation(record.payload);
            finalReceived = true;
          }
          return;
        }
        if (type === "done") {
          doneReceived = true;
          return;
        }
        if (typeof record.content === "string") {
          appendAssistantToken(record.content as string);
          return;
        }
        if (typeof record.token === "string") {
          appendAssistantToken(record.token as string);
        }
      };

      const processEvent = (rawEvent: string) => {
        if (!rawEvent.trim()) return;
        const lines = rawEvent.split("\n");
        const dataLines: string[] = [];
        for (const line of lines) {
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        }
        if (dataLines.length === 0) {
          return;
        }
        const payload = dataLines.join("\n");
        const parsed = payload ? parseJsonSafe(payload) : undefined;
        handleParsedData(parsed);
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffered += decoder.decode(value, { stream: true });
        let boundary = buffered.indexOf("\n\n");
        while (boundary !== -1) {
          const eventChunk = buffered.slice(0, boundary);
          buffered = buffered.slice(boundary + 2);
          processEvent(eventChunk);
          boundary = buffered.indexOf("\n\n");
        }
      }

      if (buffered.trim()) {
        processEvent(buffered);
      }

      if (!finalReceived) {
        throw new Error("Streaming response missing final payload.");
      }
      if (!doneReceived) {
        console.warn("[Chat] Streaming finished without explicit done event.");
      }

      const endedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
      console.info(`[Chat] Query completed in ${Math.round(endedAt - startedAt)}ms`);
    };

    try {
      await streamResponse();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to send message.";
      if (isMountedRef.current) {
        setErrorMessage(message);
      }
      setLastResponse(null);
      const defaultDetail = "Unable to fetch answer right now—please try again shortly.";
      let resolved = settlePendingAsError(trimmed, defaultDetail);
      if (!resolved && currentConversation) {
        injectLocalErrorMessage(currentConversation, trimmed, defaultDetail);
        resolved = true;
      }
      if (!resolved) {
        console.error("Unable to capture failed message locally.");
      }
      pendingMessageRef.current = null;
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };

  const handleSummarize = async () => {
    if (!currentConversation || summaryPending) return;
    setSummaryPending(true);
    setSummaryError(null);
    try {
      const data = await summarizeConversation(currentConversation.id);
      setSummaryData(data);
      setSummaryView("concise");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to summarise conversation.";
      setSummaryError(message);
    } finally {
      setSummaryPending(false);
    }
  };

  const handleExport = async (messageId: string, tableIndex: number) => {
    if (!currentConversation) return;
    setExportError(null);
    setExportLoading({ messageId, tableIndex });
    try {
      const result = await exportMessage(currentConversation.id, messageId, { tableIndex });
      if (result.delivery === "download" && result.metadata) {
        const url = `${API_BASE_URL}/api/exports/${result.metadata.token}`;
        window.open(url, "_blank", "noopener");
      } else if (result.detail) {
        setExportError({ messageId, tableIndex, message: result.detail });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to export CSV.";
      setExportError({ messageId, tableIndex, message });
    } finally {
      setExportLoading(null);
    }
  };

  const openEmailExport = (
    messageId: string,
    tableIndex: number,
    label: string,
    options?: { csvOnly?: boolean },
  ) => {
    const allowedShareOptions: ShareOptionValue[] | undefined = options?.csvOnly ? ["csv"] : undefined;
    const initialShareOption: ShareOptionValue = allowedShareOptions?.[0] ?? "csv";
    setEmailError(null);
    setEmailFeedback(null);
    setEmailModal({
      mode: "result",
      email: "",
      messageId,
      tableIndex,
      label,
      shareOption: initialShareOption,
      allowedShareOptions,
    });
  };

  const openSummaryEmail = () => {
    if (!summaryData) return;
    setEmailError(null);
    setEmailFeedback(null);
    setEmailModal({ mode: "summary", email: "", variant: summaryView });
  };

  const closeEmailModal = () => {
    setEmailModal(null);
    setEmailError(null);
    setEmailFeedback(null);
    setEmailSending(false);
  };

  const openSlackTargets = (targets?: Record<string, string>) => {
    if (!targets) return;
    if (targets.web) {
      window.open(targets.web, "_blank", "noopener");
    }
    if (targets.app) {
      window.setTimeout(() => {
        window.open(targets.app, "_blank");
      }, 150);
    }
  };

  const handleEmailSubmit = async () => {
    if (!currentConversation || !emailModal || !emailModal.email.trim()) return;
    setEmailSending(true);
    setEmailError(null);
    try {
      if (emailModal.mode === "summary") {
        const variant = emailModal.variant ?? "concise";
        const response = await sendSummaryEmail(currentConversation.id, emailModal.email.trim(), variant);
        setEmailFeedback(response.detail);
      } else {
        const response = await exportMessage(currentConversation.id, emailModal.messageId!, {
          tableIndex: emailModal.tableIndex ?? 0,
          delivery: "email",
          email: emailModal.email.trim(),
          emailMode: emailModal.shareOption ?? "csv",
        });
        setEmailFeedback(response.detail ?? `Email sent to ${emailModal.email.trim()}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to send email.";
      setEmailError(message);
    } finally {
      setEmailSending(false);
    }
  };

  const beginTitleEdit = () => {
    if (!currentConversation) return;
    setTitleDraft(currentConversation.title || "");
    setTitleEditing(true);
    setTitleError(null);
  };

  const cancelTitleEdit = () => {
    setTitleEditing(false);
    setTitleError(null);
    setTitleDraft(currentConversation?.title || "");
  };

  const saveConversationTitle = async () => {
    if (!currentConversation) return;
    setTitleSaving(true);
    setTitleError(null);
    try {
      const updated = await updateConversationTitle(currentConversation.id, titleDraft.trim());
      syncConversation(updated);
      setTitleEditing(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to rename conversation.";
      setTitleError(message);
    } finally {
      setTitleSaving(false);
    }
  };

  useEffect(() => {
    if (!focusMessageId || !currentConversation) {
      return;
    }
    clearHighlight(false);
    autoScrollLockRef.current = true;
    let attempts = 0;
    const maxAttempts = 25;
    const messages = currentConversation.messages || [];

    const scrollIntoView = () => {
      const container = messagesContainerRef.current;
      const el = document.getElementById(`msg-${focusMessageId}`);
      if (container && el) {
        const containerRect = container.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        const offset = elRect.top - containerRect.top;
        const targetTop = container.scrollTop + offset - container.clientHeight * 0.25;
        container.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });

        const idx = messages.findIndex((message) => message.id === focusMessageId);
        const ids = new Set<string>();
        if (idx !== -1) {
          const targetMessage = messages[idx];
          ids.add(targetMessage.id);
          if (targetMessage.role === "user") {
            const answer = messages[idx + 1];
            if (answer && answer.role === "assistant") {
              ids.add(answer.id);
            }
          } else if (targetMessage.role === "assistant") {
            const question = messages[idx - 1];
            if (question && question.role === "user") {
              ids.add(question.id);
            }
          }
        } else {
          ids.add(focusMessageId);
        }
        setHighlightedIds(Array.from(ids));

        if (highlightFadeTimeoutRef.current) {
          window.clearTimeout(highlightFadeTimeoutRef.current);
        }
        highlightFadeTimeoutRef.current = window.setTimeout(() => {
          setHighlightedIds([]);
          highlightFadeTimeoutRef.current = null;
        }, 2200);

        if (highlightCleanupRef.current) {
          highlightCleanupRef.current();
          highlightCleanupRef.current = null;
        }

        const handleInteraction = () => clearHighlight(true);
        container.addEventListener("scroll", handleInteraction, { once: true });
        window.addEventListener("pointerdown", handleInteraction, { once: true });
        window.addEventListener("keydown", handleInteraction, { once: true });
        highlightCleanupRef.current = () => {
          container.removeEventListener("scroll", handleInteraction);
          window.removeEventListener("pointerdown", handleInteraction);
          window.removeEventListener("keydown", handleInteraction);
        };

        setFocusMessageId(undefined);
        return;
      }
      if (attempts < maxAttempts) {
        attempts += 1;
        scrollRetryRef.current = window.setTimeout(scrollIntoView, 200);
      } else {
        console.warn("Could not find message element for focus", focusMessageId);
        clearHighlight(true);
        setFocusMessageId(undefined);
      }
    };

    scrollIntoView();
    return () => {
      if (scrollRetryRef.current) {
        window.clearTimeout(scrollRetryRef.current);
        scrollRetryRef.current = null;
      }
    };
  }, [
    focusMessageId,
    currentConversation?.id,
    currentConversation?.messages?.length,
    setFocusMessageId,
    clearHighlight,
  ]);

  useEffect(() => {
    return () => {
      clearHighlight(true);
    };
  }, [clearHighlight]);

  useEffect(() => {
    if (!currentConversation) return;
    const latest = [...currentConversation.messages]
      .filter((message) => message.role === "assistant" && message.payload?.action)
      .pop();
    if (!latest || !latest.payload?.action) return;
    if (actionHandledRef.current.has(latest.id)) return;
    actionHandledRef.current.add(latest.id);
    if (latest.payload.action === "open_help_modal") {
      setHelpOpen(true);
    }
    if (latest.payload.action === "open_slackbot") {
      openSlackTargets(latest.payload.targets as Record<string, string> | undefined);
    }
    if (latest.payload.action === "open_export") {
      navigate("/data-export");
    }
  }, [currentConversation, setHelpOpen, navigate]);

  useEffect(() => {
    setEmailModal((prev) => (prev && prev.mode === "summary" ? { ...prev, variant: summaryView } : prev));
  }, [summaryView]);

  if (!currentConversation) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="rounded-2xl border border-slate-200 bg-white/95 shadow-lg px-6 py-5 text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">
          {WELCOME_MESSAGE}
        </div>
        <ChatInput disabled onSend={async () => undefined} />
      </div>
    );
  }

  const summaryText =
    summaryData && (summaryView === "concise" ? summaryData.concise : summaryData.detailed);
  const renderSummaryContent = () => {
    if (!summaryData) return null;
    const renderStructuredList = (
      items?: string[],
      { allowHeadings = true }: SummaryListOptions = {},
    ) => {
      const structured = buildStructuredSummaryItems(items, allowHeadings);
      if (!structured.length) return null;
      const indentClass = (depth: number) => {
        if (depth <= 0) return "pl-1";
        if (depth === 1) return "pl-5";
        return "pl-8";
      };
      return (
        <div className="mt-1 space-y-1 text-sm text-slate-700">
          {structured.map((entry, idx) => {
            if (entry.kind === "heading") {
              const headingClass =
                entry.depth > 0
                  ? "text-sm font-semibold text-slate-800"
                  : "text-xs font-semibold uppercase tracking-widest text-slate-500";
              const spacingClass = idx === 0 ? "pt-1" : "pt-3";
              return (
                <div key={`heading-${idx}-${entry.text}`} className={`${headingClass} ${spacingClass}`}>
                  {entry.text}
                </div>
              );
            }
            const indent = indentClass(entry.depth);
            return (
              <div key={`bullet-${idx}-${entry.text}`} className={`flex items-start gap-2 ${indent}`}>
                <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-slate-400" />
                <span className="leading-relaxed">{entry.text}</span>
              </div>
            );
          })}
        </div>
      );
    };
    if (summaryView === "concise" && summaryData.concise_sections?.length) {
      return (
        <div className="space-y-4">
          {summaryData.concise_sections.map((section, index) => (
            <div
              key={`${section.title}-${index}`}
              className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{section.title}</p>
              <ul className="mt-2 list-disc pl-5 text-sm text-slate-700 space-y-1">
                {(section.items ?? []).map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      );
    }
    if (summaryView === "detailed" && summaryData.detailed_topics?.length) {
      return (
        <div className="space-y-4">
          {summaryData.detailed_topics.map((topic, index) => {
            const tables =
              topic.answer_tables && topic.answer_tables.length
                ? topic.answer_tables
                : topic.answer_table
                ? [topic.answer_table]
                : [];
            return (
              <div
                key={`${topic.title}-${index}`}
                className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm space-y-3"
              >
                <div className="text-sm font-semibold text-slate-800">{topic.title}</div>
                {topic.question_items?.length ? (
                  <div>
                    <p className="text-xs uppercase tracking-widest text-slate-500">Question</p>
                    {renderStructuredList(topic.question_items, { allowHeadings: false })}
                  </div>
                ) : topic.question ? (
                  <p className="text-sm text-slate-700">{topic.question}</p>
                ) : null}
                {topic.answer_items?.length ? (
                  <div>
                    <p className="text-xs uppercase tracking-widest text-slate-500">Answer</p>
                    {renderStructuredList(topic.answer_items)}
                  </div>
                ) : topic.answer ? (
                  <p className="text-sm text-slate-700 whitespace-pre-line">{topic.answer}</p>
                ) : null}
                {tables.length > 0 &&
                  tables.map((table, tableIdx) => (
                    <div
                      key={`${topic.title}-table-${tableIdx}`}
                      className="overflow-x-auto rounded-xl border border-slate-200 bg-white"
                    >
                      {tables.length > 1 && (
                        <div className="px-3 pt-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                          Table {tableIdx + 1}
                        </div>
                      )}
                      <table className="w-full text-sm text-left text-slate-700">
                        <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-500">
                          <tr>
                            {table.headers.map((header, idx) => (
                              <th key={idx} className="px-3 py-2 border-b border-slate-200">
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {table.rows.map((row, rowIndex) => (
                            <tr key={rowIndex} className="odd:bg-white even:bg-slate-50">
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} className="px-3 py-2 border-b border-slate-100">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
              </div>
            );
          })}
        </div>
      );
    }
    return <p className="text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">{summaryText}</p>;
  };
  const telemetryInfo = lastResponse?.telemetry as { latency_ms?: number; total_latency_s?: number } | undefined;
  const latencyMs = telemetryInfo?.latency_ms;

  const filteredEmailShareOptions =
    emailModal && emailModal.allowedShareOptions && emailModal.allowedShareOptions.length > 0
      ? EMAIL_SHARE_OPTIONS.filter((option) =>
          emailModal.allowedShareOptions?.includes(option.value),
        )
      : EMAIL_SHARE_OPTIONS;

  const emailModalDescription = (() => {
    if (!emailModal) return "";
    if (emailModal.mode === "summary") {
      return `You’re sharing the ${emailModal.variant ?? "concise"} summary snapshot.`;
    }
    if (
      emailModal.allowedShareOptions &&
      emailModal.allowedShareOptions.length === 1 &&
      emailModal.allowedShareOptions[0] === "csv"
    ) {
      return `Attach the “${emailModal.label ?? "Result"}” table as a CSV export.`;
    }
    return `Include the generated SQL, attach the “${emailModal.label ?? "Result"}” table as CSV, or send both.`;
  })();

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div className="flex flex-col gap-1">
          {titleEditing ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  value={titleDraft}
                  onChange={(event) => setTitleDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      void saveConversationTitle();
                    }
                    if (event.key === "Escape") {
                      event.preventDefault();
                      cancelTitleEdit();
                    }
                  }}
                  maxLength={200}
                  className="min-w-[240px] rounded-2xl border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-800 shadow-sm focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-100"
                  placeholder="Add a chat title"
                  autoFocus
                />
                <button
                  onClick={() => void saveConversationTitle()}
                  disabled={titleSaving}
                  className="rounded-full bg-primary-500 px-4 py-1.5 text-sm font-semibold text-white shadow hover:bg-primary-600 disabled:opacity-60"
                >
                  {titleSaving ? "Saving…" : "Save"}
                </button>
                <button
                  onClick={cancelTitleEdit}
                  className="rounded-full border border-slate-300 px-4 py-1.5 text-sm font-semibold text-slate-600 hover:border-slate-400"
                >
                  Cancel
                </button>
              </div>
              <p className="text-xs text-slate-500">
                Leave blank to auto-name the chat from the first user message.
              </p>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-semibold text-slate-800">
                {currentConversation.title || "New conversation"}
              </h1>
              <button
                onClick={beginTitleEdit}
                className="inline-flex items-center justify-center rounded-full border border-slate-300 p-2 text-slate-500 hover:border-primary-300 hover:text-primary-600"
                title="Rename chat"
                aria-label="Rename chat"
              >
                ✐
              </button>
            </div>
          )}
          {titleError && <p className="text-sm text-red-500">{titleError}</p>}
        </div>
        <div className="flex items-center gap-3">
          {summaryError && <span className="text-sm text-red-500">{summaryError}</span>}
          <button
            onClick={handleSummarize}
            disabled={summaryPending}
            className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600 disabled:opacity-60"
          >
            {summaryPending ? "Summarising…" : "Summarise"}
          </button>
        </div>
      </div>

      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto pr-3 space-y-4">
        {currentConversation.messages.length === 0 ? (
          <div className="rounded-2xl border border-slate-200 bg-white/95 shadow-lg px-6 py-5 text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">
            {WELCOME_MESSAGE}
          </div>
        ) : (
          currentConversation.messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              highlighted={highlightedIds.includes(message.id)}
              exportLoading={exportLoading}
              exportError={exportError}
              onExport={(tableIndex) => handleExport(message.id, tableIndex)}
              onEmailExport={(tableIndex, label, options) => openEmailExport(message.id, tableIndex, label, options)}
            />
          ))
        )}
      </div>

      {!loading && typeof latencyMs === "number" && (
        <div className="mt-3 inline-flex flex-col gap-1 rounded-xl border border-slate-200 bg-white/85 px-4 py-3 text-xs text-slate-600 shadow-sm self-start">
          <span>
            <strong className="font-semibold text-slate-700">Latency:</strong> {Math.round(latencyMs)} ms
          </span>
        </div>
      )}


      <ChatInput
        disabled={loading || userMessageCount >= 50}
        onSend={handleSend}
        error={errorMessage ?? (userMessageCount >= 50 ? "Limit reached—open a new chat." : null)}
      />

      {summaryData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-6">
          <div className="w-full max-w-2xl rounded-3xl bg-white border border-slate-200 shadow-2xl p-6 space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h3 className="text-xl font-semibold text-slate-800">Conversation summary</h3>
                <p className="text-sm text-slate-500">
                  Toggle between concise highlights and the full question → answer recap.
                </p>
              </div>
              <button
                onClick={() => setSummaryData(null)}
                className="text-slate-400 hover:text-slate-600 text-lg"
                aria-label="Close summary"
              >
                ✕
              </button>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex rounded-full bg-slate-100 p-1">
                <button
                  onClick={() => setSummaryView("concise")}
                  className={`px-4 py-1 rounded-full text-sm font-medium transition ${
                    summaryView === "concise"
                      ? "bg-white shadow text-primary-600"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  Concise
                </button>
                <button
                  onClick={() => setSummaryView("detailed")}
                  className={`px-4 py-1 rounded-full text-sm font-medium transition ${
                    summaryView === "detailed"
                      ? "bg-white shadow text-primary-600"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  Detailed
                </button>
              </div>
              <button
                onClick={openSummaryEmail}
                className="rounded-full border border-primary-200 px-4 py-1.5 text-sm font-medium text-primary-600 hover:bg-primary-50"
              >
                Email this view
              </button>
            </div>
            <div className="max-h-[60vh] overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50/70 p-5 text-sm leading-relaxed text-slate-700">
              {renderSummaryContent()}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setSummaryData(null)}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {emailModal && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/50 px-6">
          <div className="w-full max-w-md rounded-2xl bg-white border border-slate-200 shadow-2xl p-6 space-y-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-800">
                  {emailModal.mode === "summary" ? "Send summary via email" : "Share SQL / CSV via email"}
                </h3>
                {emailModalDescription && <p className="text-sm text-slate-500">{emailModalDescription}</p>}
              </div>
              <button onClick={closeEmailModal} className="text-slate-400 hover:text-slate-600 text-lg" aria-label="Close email modal">
                ✕
              </button>
            </div>
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-600">
                Recipient email
                <input
                  type="email"
                  value={emailModal.email}
                  onChange={(event) =>
                    setEmailModal((prev) => (prev ? { ...prev, email: event.target.value } : prev))
                  }
                  className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-inner focus:border-primary-400 focus:outline-none"
                  placeholder="name@example.com"
                />
              </label>
              {emailModal.mode === "result" && filteredEmailShareOptions.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-slate-600">Email contents</p>
                  <div className="mt-2 space-y-2">
                    {filteredEmailShareOptions.map((option) => (
                      <label
                        key={option.value}
                        className={`flex items-start gap-3 rounded-2xl border px-3 py-2 text-sm transition ${
                          emailModal.shareOption === option.value
                            ? "border-primary-400 bg-primary-50/60"
                            : "border-slate-200 bg-slate-50/70"
                        }`}
                      >
                        <input
                          name="email-share-option"
                          type="radio"
                          checked={emailModal.shareOption === option.value}
                          onChange={() =>
                            setEmailModal((prev) =>
                              prev ? { ...prev, shareOption: option.value } : prev,
                            )
                          }
                          className="mt-1 accent-primary-500"
                        />
                        <div>
                          <div className="font-semibold text-slate-700">{option.label}</div>
                          <p className="text-xs text-slate-500">{option.description}</p>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              )}
              {emailFeedback && <p className="text-sm text-emerald-600">{emailFeedback}</p>}
              {emailError && <p className="text-sm text-red-500">{emailError}</p>}
            </div>
            <div className="flex justify-end gap-3">
              <button
                onClick={closeEmailModal}
                className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                onClick={handleEmailSubmit}
                disabled={emailSending || !emailModal.email.trim()}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600 disabled:opacity-60"
              >
                {emailSending ? "Sending…" : "Send email"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
