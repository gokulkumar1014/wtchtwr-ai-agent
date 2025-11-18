import type { Conversation } from "@/store/useChat";

const DEBUG = import.meta.env.VITE_DEBUG === "true";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  if (DEBUG) {
    console.info("[API_REQUEST]", {
      url: typeof input === "string" ? input : input.toString(),
      method: init?.method ?? "GET",
    });
  }
  const started = typeof performance !== "undefined" ? performance.now() : Date.now();
  const response = await fetch(input, {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });
  const ended = typeof performance !== "undefined" ? performance.now() : Date.now();
  if (DEBUG) {
    console.info("[API_RESPONSE]", {
      url: typeof input === "string" ? input : input.toString(),
      status: response.status,
      durationMs: Math.round(ended - started),
    });
  }
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = await response.json();
      detail = (data && (data.detail || data.message)) ?? detail;
    } catch (err) {
      const text = await response.text();
      detail = text || detail;
    }
    throw new Error(detail);
  }
  const data = (await response.json()) as T;
  if (DEBUG) {
    const keys =
      data && typeof data === "object" && !Array.isArray(data) ? Object.keys(data as Record<string, unknown>) : [];
    console.info("[API_RESPONSE_DATA]", {
      url: typeof input === "string" ? input : input.toString(),
      keys,
    });
  }
  return data;
}

export async function listConversations(): Promise<Conversation[]> {
  return request<Conversation[]>(`${API_BASE_URL}/api/conversations`);
}

export async function createConversation(title?: string): Promise<Conversation> {
  return request<Conversation>(`${API_BASE_URL}/api/conversations`, {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function getConversation(id: string): Promise<Conversation> {
  return request<Conversation>(`${API_BASE_URL}/api/conversations/${id}`);
}

export async function sendMessage(
  conversationId: string,
  message: string,
  options?: { stream?: boolean },
): Promise<Conversation> {
  return request<Conversation>(`${API_BASE_URL}/api/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ role: "user", content: message, stream: options?.stream ?? false }),
  });
}

export interface ConversationSummaryPayload {
  conversation_id: string;
  concise: string;
  detailed: string;
  concise_sections?: Array<{ title: string; items: string[] }>;
  detailed_topics?: Array<{
    title: string;
    question?: string;
    question_items?: string[];
    answer?: string;
    answer_items?: string[];
    answer_table?: { headers: string[]; rows: string[][] };
    answer_tables?: Array<{ headers: string[]; rows: string[][] }>;
  }>;
}

export async function summarizeConversation(conversationId: string): Promise<ConversationSummaryPayload> {
  return request<ConversationSummaryPayload>(`${API_BASE_URL}/api/conversations/${conversationId}/summary`, {
    method: "POST",
  });
}

export async function deleteConversation(conversationId: string): Promise<Conversation[]> {
  return request<Conversation[]>(`${API_BASE_URL}/api/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

export async function deleteMessage(conversationId: string, messageId: string): Promise<Conversation> {
  return request<Conversation>(`${API_BASE_URL}/api/conversations/${conversationId}/messages/${messageId}`, {
    method: "DELETE",
  });
}

export interface ExportMetadata {
  token: string;
  format: string;
  rows: number;
  expires_at: string;
  filename: string;
  session_only: boolean;
}

export interface ExportActionResponse {
  delivery: "download" | "email";
  metadata?: ExportMetadata;
  detail?: string;
}

export interface DataExplorerColumn {
  name: string;
  data_type: string;
  description?: string;
}

export interface DataExplorerTableMeta {
  name: string;
  columns: DataExplorerColumn[];
}

export interface DataExplorerSchemaResponse {
  tables: DataExplorerTableMeta[];
}

export interface DataExplorerColumnSelection {
  table: string;
  column: string;
}

export interface DataExplorerFilter {
  table: string;
  column: string;
  operator?: string;
  value?: string | number | string[];
}

export interface DataExplorerSort {
  table: string;
  column: string;
  direction?: "asc" | "desc";
}

export interface DataExplorerQueryRequest {
  tables: string[];
  columns: DataExplorerColumnSelection[];
  filters?: DataExplorerFilter[];
  sort?: DataExplorerSort[];
  joins?: Array<{ left_table: string; left_column: string; right_table: string; right_column: string }>;
  limit?: number;
}

export interface DataExplorerQueryResponse {
  sql: string;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  limit: number;
  tables: string[];
}

export interface ExportMessageOptions {
  tableIndex?: number;
  delivery?: "download" | "email";
  email?: string;
  emailMode?: "sql" | "csv" | "both";
}

export async function exportMessage(
  conversationId: string,
  messageId: string,
  options?: ExportMessageOptions,
): Promise<ExportActionResponse> {
  const payload = {
    table_index: options?.tableIndex ?? 0,
    delivery: options?.delivery ?? "download",
    email: options?.email,
    email_mode: options?.emailMode ?? "csv",
  };
  return request<ExportActionResponse>(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/${messageId}/export`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function sendSummaryEmail(
  conversationId: string,
  email: string,
  variant: "concise" | "detailed",
): Promise<{ detail: string; variant: "concise" | "detailed" }> {
  return request<{ detail: string; variant: "concise" | "detailed" }>(
    `${API_BASE_URL}/api/conversations/${conversationId}/summary/email`,
    {
      method: "POST",
      body: JSON.stringify({ email, variant }),
    },
  );
}

export async function fetchDataExplorerSchema(): Promise<DataExplorerSchemaResponse> {
  return request<DataExplorerSchemaResponse>(`${API_BASE_URL}/api/data-explorer/schema`);
}

export async function runDataExplorerQuery(body: DataExplorerQueryRequest): Promise<DataExplorerQueryResponse> {
  return request<DataExplorerQueryResponse>(`${API_BASE_URL}/api/data-explorer/query`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function exportDataExplorer(
  body: DataExplorerQueryRequest & { delivery: "download" | "email"; email?: string },
): Promise<ExportActionResponse> {
  return request<ExportActionResponse>(`${API_BASE_URL}/api/data-explorer/export`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function fetchColumnValues(table: string, column: string): Promise<{ values: string[] }> {
  return request<{ values: string[] }>(`${API_BASE_URL}/api/data-explorer/tables/${encodeURIComponent(table)}/columns/${encodeURIComponent(column)}/values`);
}

export interface DashboardRange {
  min?: number;
  max?: number;
}

export interface DashboardFilterOptions {
  neighborhoodGroups: string[];
  neighborhoods: string[];
  propertyTypes: string[];
  roomTypes: string[];
  hostNames: string[];
  bathroomDetails: string[];
  ranges: {
    accommodates: DashboardRange;
    bathrooms: DashboardRange;
    bedrooms: DashboardRange;
    beds: DashboardRange;
    price: DashboardRange;
  };
}

export interface DashboardFiltersPayload {
  neighborhoodGroups: string[];
  neighborhoods: string[];
  propertyTypes: string[];
  roomTypes: string[];
  accommodates: DashboardRange;
  bathrooms: DashboardRange;
  bedrooms: DashboardRange;
  beds: DashboardRange;
  price: DashboardRange;
  hostNames: string[];
  bathroomDetails: string[];
}

export interface DashboardComparisonPayload {
  mode: "market" | "hosts";
  hosts: string[];
}

export interface DashboardRequestPayload {
  filters: DashboardFiltersPayload;
  comparison: DashboardComparisonPayload;
}

export interface DashboardMetricComparison {
  metric: string;
  label: string;
  highbury: number | null;
  comparison: number | null;
}

export interface DashboardPriceSummary {
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
}

export interface DashboardHostListingCount {
  hostName: string;
  listings: number;
}

export interface DashboardRatingBucket {
  label: string;
  highbury: number;
  comparison: number;
}

export interface DashboardSummary {
  totals: {
    highburyListings: number;
    comparisonListings: number;
  };
  occupancyCards: DashboardMetricComparison[];
  revenueCards: DashboardMetricComparison[];
  reviewScores: DashboardMetricComparison[];
  priceSummary: {
    highbury: DashboardPriceSummary;
    comparison: DashboardPriceSummary;
  };
  ratingSummary: {
    highburyAverage: number | null;
    comparisonAverage: number | null;
    distribution: DashboardRatingBucket[];
  };
  hostCounts: {
    highbury: DashboardHostListingCount[];
    comparison: DashboardHostListingCount[];
    combined: DashboardHostListingCount[];
  };
}

export type DashboardListingGroup = "Highbury" | "Comparison";

export interface DashboardMapListing {
  listingId: number;
  lat: number;
  lng: number;
  group: DashboardListingGroup;
  hostName: string;
  propertyName: string | null;
  neighborhood: string | null;
  neighborhoodGroup: string | null;
  propertyType: string | null;
  roomType: string | null;
  price: number | null;
  occupancyRate90: number | null;
  reviewScore: number | null;
}

export interface DashboardAvailableFilters {
  neighborhoodGroups: string[];
  neighborhoods: string[];
  propertyTypes: string[];
  roomTypes: string[];
  hostNames: string[];
  bathroomDetails: string[];
}

export interface DashboardViewResponse {
  summary: DashboardSummary;
  map: {
    listings: DashboardMapListing[];
    total: number;
    rendered: number;
    comparisonSampled: number;
  };
  availableFilters: DashboardAvailableFilters;
}

export async function fetchDashboardFilters(): Promise<DashboardFilterOptions> {
  return request<DashboardFilterOptions>(`${API_BASE_URL}/api/dashboard/filters`);
}

export async function fetchDashboardView(body: DashboardRequestPayload): Promise<DashboardViewResponse> {
  return request<DashboardViewResponse>(`${API_BASE_URL}/api/dashboard/view`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
