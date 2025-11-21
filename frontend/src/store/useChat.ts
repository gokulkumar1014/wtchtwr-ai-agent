import { create } from "zustand";
import { normalizeConversation } from "@/lib/messages";

export interface TablePayload {
  name: string;
  columns: string[];
  data: Record<string, unknown>[];
  preview?: string;
  row_count?: number;
  source?: string;
  sql?: string;
}

export interface ThinkingStep {
  phase: string;
  title: string;
  detail: string;
  meta?: Record<string, unknown>;
  elapsed_ms?: number;
}

export interface AssistantPayload {
  sql?: string;
  params?: string[];
  tables?: TablePayload[];
  summary?: string;
  answer_text?: string;
  question?: string;
  export?: Record<string, unknown>;
  template_id?: string;
  row_count?: number;
  duration_ms?: number;
  response_type?: "sql" | "rag" | "sentiment" | "hybrid" | string;
  action?: string;
  targets?: Record<string, string>;
  markdown_table?: string;
  pipeline?: string;
  policy?: string;
  telemetry?: Record<string, unknown>;
  intent?: string;
  expansion_report?: string;
  expansion_sources?: Array<{
    url?: string;
    title?: string;
    text?: string;
    score?: number | string;
  }>;
  sentiment_analytics?: {
    positive: number;
    neutral: number;
    negative: number;
    total_reviews?: number;
    dominant_label?: string;
    dominant_label_display?: string;
    overall_sentiment?: string;
    average_sentiment_strength?: number;
  };
  rag_snippets?: Array<
    | string
    | {
        snippet?: string;
        text?: string;
        month?: string;
        year?: string | number;
        neighbourhood?: string;
        borough?: string;
        neighbourhood_group?: string;
        listing_name?: string;
        listing_id?: string | number;
        comment_id?: string | number;
        commentId?: string | number;
        commentID?: string | number;
        review_id?: string | number;
        reviewId?: string | number;
        citation?: string;
        sentiment_label?: string;
        compound?: number | string;
        positive?: number | string;
        neutral?: number | string;
        negative?: number | string;
        is_highbury?: boolean | string;
      }
  >;
  thinking_trace?: ThinkingStep[];
  portfolio_triage?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content?: string;
  nl_summary?: string;
  payload?: AssistantPayload;
  timestamp: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

interface ChatState {
  conversations: Conversation[];
  activeConversation?: Conversation;
  loading: boolean;
  setConversations: (conversations: Conversation[]) => void;
  setActiveConversation: (conversation: Conversation) => void;
  setLoading: (value: boolean) => void;
  focusMessageId?: string;
  setFocusMessageId: (id?: string) => void;
  removeConversation: (id: string) => void;
  upsertConversation: (conversation: Conversation) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversation: undefined,
  loading: false,
  setConversations: (conversations) =>
    set((state) => {
      const normalizedList = conversations.map((conversation) => normalizeConversation(conversation));
      const activeConversation =
        state.activeConversation &&
        normalizedList.find((conversation) => conversation.id === state.activeConversation?.id);
      return {
        conversations: normalizedList,
        activeConversation: activeConversation ?? state.activeConversation,
      };
    }),
  setActiveConversation: (conversation) =>
    set((state) => {
      const normalized = normalizeConversation(conversation);
      const existing =
        state.conversations.find((conv) => conv.id === normalized.id) ?? normalized;
      return { activeConversation: existing };
    }),
  setLoading: (value) => set({ loading: value }),
  setFocusMessageId: (id) => set({ focusMessageId: id }),
  removeConversation: (id) =>
    set((state) => ({
      conversations: state.conversations.filter((convo) => convo.id !== id),
      activeConversation:
        state.activeConversation && state.activeConversation.id === id
          ? undefined
          : state.activeConversation,
    })),
  upsertConversation: (conversation) =>
    set((state) => {
      const normalized = normalizeConversation(conversation);
      const others = state.conversations.filter((c) => c.id !== conversation.id);
      const conversations = [normalized, ...others];
      const activeConversation =
        state.activeConversation && state.activeConversation.id === conversation.id
          ? normalized
          : state.activeConversation;
      return { conversations, activeConversation };
    }),
}));
