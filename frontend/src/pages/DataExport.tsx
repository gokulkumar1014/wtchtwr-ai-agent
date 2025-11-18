import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  API_BASE_URL,
  DataExplorerQueryRequest,
  DataExplorerQueryResponse,
  DataExplorerSchemaResponse,
  DataExplorerTableMeta,
  ExportActionResponse,
  exportDataExplorer,
  fetchColumnValues,
  fetchDataExplorerSchema,
  runDataExplorerQuery,
} from "@/lib/api";

interface FilterRow {
  id: string;
  table: string;
  column: string;
  operator: string;
  value: string;
  valueB?: string;
  selectedValues?: string[];
}

interface SortRow {
  id: string;
  table: string;
  column: string;
  direction: "asc" | "desc";
}

const OPERATORS = [
  { value: "equals", label: "Equals" },
  { value: "not_equals", label: "Does not equal" },
  { value: "gt", label: "Greater than" },
  { value: "gte", label: "Greater than or equal" },
  { value: "lt", label: "Less than" },
  { value: "lte", label: "Less than or equal" },
  { value: "between", label: "Between" },
  { value: "contains", label: "Contains" },
  { value: "not_contains", label: "Does not contain" },
  { value: "starts_with", label: "Begins with" },
  { value: "not_starts_with", label: "Does not begin with" },
  { value: "ends_with", label: "Ends with" },
  { value: "not_ends_with", label: "Does not end with" },
];

const LOAD_CHUNK = 60;
const DEFAULT_LIMIT = 200;
const TABLE_DISPLAY_NAMES: Record<string, string> = {
  amenities_norm: "Amenities (Normalised)",
  amenities_raw: "Amenities (Raw)",
  listings_cleaned: "Listings (All Market)",
  reviews_enriched: "Reviews (Enriched)",
  text_extract: "Text Extract",
  highbury_listings: "Highbury Listings",
};

const uuid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(16).slice(2);

export const DataExportPage: React.FC = () => {
  const navigate = useNavigate();
  const [schema, setSchema] = useState<DataExplorerTableMeta[]>([]);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [loadingSchema, setLoadingSchema] = useState(true);

  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<Record<string, string[]>>({});
  const [filters, setFilters] = useState<FilterRow[]>([]);
  const [sorts, setSorts] = useState<SortRow[]>([]);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);

  const [preview, setPreview] = useState<DataExplorerQueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [autoRefreshing, setAutoRefreshing] = useState(false);

  const [emailModalOpen, setEmailModalOpen] = useState(false);
  const [emailAddress, setEmailAddress] = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [emailStatus, setEmailStatus] = useState<string | null>(null);
  const [emailError, setEmailError] = useState<string | null>(null);
  const [expandedTables, setExpandedTables] = useState<Record<string, boolean>>({});

  const [sqlModalOpen, setSqlModalOpen] = useState(false);

  const [filterPanel, setFilterPanel] = useState<{
    id: string;
    table: string;
    column: string;
    search: string;
    visible: number;
    pending: string[];
  } | null>(null);
  const [filterOptions, setFilterOptions] = useState<Record<string, string[]>>({});

  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const response: DataExplorerSchemaResponse = await fetchDataExplorerSchema();
        setSchema(response.tables);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to load schema.";
        setSchemaError(message);
      } finally {
        setLoadingSchema(false);
      }
    })();
  }, []);

  const schemaMap = useMemo(() => {
    const map: Record<string, DataExplorerTableMeta> = {};
    schema.forEach((table) => {
      map[table.name] = table;
    });
    return map;
  }, [schema]);

  const hasColumnSelection = useMemo(
    () => selectedTables.some((table) => (selectedColumns[table]?.length ?? 0) > 0),
    [selectedTables, selectedColumns],
  );

  const previewMinWidth = useMemo(() => {
    if (!preview) return 800;
    return Math.max(preview.columns.length * 160, 800);
  }, [preview]);

  const buildPayload = (): DataExplorerQueryRequest | null => {
    if (!selectedTables.length) return null;
    const columns = selectedTables.flatMap((table) => (selectedColumns[table] || []).map((column) => ({ table, column })));
    if (!columns.length) return null;

    const filtersPayload = filters
      .map((row) => {
        if (!row.table || !row.column) return null;
        if (row.operator === "between") {
          if (!row.value.trim() || !row.valueB?.trim()) return null;
          return { table: row.table, column: row.column, operator: "between", value: [row.value, row.valueB] };
        }
        if (row.selectedValues && row.selectedValues.length) {
          return { table: row.table, column: row.column, operator: "in", value: row.selectedValues };
        }
        if (!row.value.trim()) return null;
        return { table: row.table, column: row.column, operator: row.operator, value: row.value };
      })
      .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));

    const sortPayload = sorts
      .filter((row) => row.table && row.column)
      .map((row) => ({ table: row.table, column: row.column, direction: row.direction }));

    return {
      tables: selectedTables,
      columns,
      filters: filtersPayload,
      sort: sortPayload,
      limit,
    };
  };

  const runPreview = async (auto = false) => {
    const payload = buildPayload();
    if (!payload) {
      if (!auto) setQueryError("Select at least one table and column to preview data.");
      setPreview(null);
      setAutoRefreshing(false);
      return;
    }
    if (auto) setAutoRefreshing(true);
    setQueryError(null);
    try {
      const response = await runDataExplorerQuery(payload);
      setPreview(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to run query.";
      setQueryError(message);
      setPreview(null);
    } finally {
      if (auto) setAutoRefreshing(false);
    }
  };

  useEffect(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    if (!selectedTables.length || !hasColumnSelection) {
      setPreview(null);
      setQueryError(null);
      return;
    }
    debounceRef.current = window.setTimeout(() => runPreview(true), 500);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [selectedTables, selectedColumns, filters, sorts, limit, hasColumnSelection]);

  const triggerDownload = async () => {
    const payload = buildPayload();
    if (!payload) {
      setQueryError("Select at least one table and column before exporting.");
      return;
    }
    setDownloading(true);
    setQueryError(null);
    try {
      const response = await exportDataExplorer({ ...payload, limit: undefined, delivery: "download" });
      if (response.metadata) {
        const url = `${API_BASE_URL}/api/exports/${response.metadata.token}`;
        window.open(url, "_blank", "noopener");
      } else if (response.detail) {
        setQueryError(response.detail);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to export CSV.";
      setQueryError(message);
    } finally {
      setDownloading(false);
    }
  };

  const handleEmailSubmit = async () => {
    if (!emailAddress.trim()) return;
    const payload = buildPayload();
    if (!payload) return;
    setEmailSending(true);
    setEmailError(null);
    setEmailStatus(null);
    try {
      const response = await exportDataExplorer({ ...payload, limit: undefined, delivery: "email", email: emailAddress.trim() });
      setEmailStatus(response.detail ?? `Data export emailed to ${emailAddress.trim()}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to send export.";
      setEmailError(message);
    } finally {
      setEmailSending(false);
    }
  };

  const openEmailModal = () => {
    const payload = buildPayload();
    if (!payload) {
      setQueryError("Select at least one table and column to send an export.");
      return;
    }
    setEmailModalOpen(true);
    setEmailAddress("");
    setEmailStatus(null);
    setEmailError(null);
  };

  const handleClearAll = () => {
    setSelectedTables([]);
    setSelectedColumns({});
    setFilters([]);
    setSorts([]);
    setPreview(null);
    setQueryError(null);
    setFilterPanel(null);
  };

  const toggleTable = (tableName: string) => {
    setSelectedTables((prev) => {
      if (prev.includes(tableName)) {
        const nextTables = prev.filter((name) => name !== tableName);
        setSelectedColumns((cols) => {
          const next = { ...cols };
          delete next[tableName];
          return next;
        });
        setFilters((rows) => rows.filter((row) => row.table !== tableName));
        setSorts((rows) => rows.filter((row) => row.table !== tableName));
        setExpandedTables((exp) => {
          const next = { ...exp };
          delete next[tableName];
          return next;
        });
        return nextTables;
      }
      setExpandedTables((exp) => ({ ...exp, [tableName]: true }));
      return [...prev, tableName];
    });
  };

  const ensureTableSelected = (tableName: string) => {
    setSelectedTables((prev) => (prev.includes(tableName) ? prev : [...prev, tableName]));
  };

  const toggleTableExpansion = (tableName: string) => {
    setExpandedTables((prev) => ({
      ...prev,
      [tableName]: !prev[tableName],
    }));
  };

  const toggleColumn = (tableName: string, column: string) => {
    ensureTableSelected(tableName);
    setExpandedTables((prev) => ({ ...prev, [tableName]: true }));
    setSelectedColumns((prev) => {
      const existing = prev[tableName] || [];
      const next = existing.includes(column) ? existing.filter((col) => col !== column) : [...existing, column];
      return { ...prev, [tableName]: next };
    });
  };

  const addFilter = () => {
    if (!selectedTables.length) return;
    const table = selectedTables[0];
    const firstColumn = schemaMap[table]?.columns[0]?.name;
    if (!firstColumn) return;
    setFilters((prev) => [...prev, { id: uuid(), table, column: firstColumn, operator: "equals", value: "" }]);
  };

  const updateFilter = (id: string, patch: Partial<FilterRow>) => {
    setFilters((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  };

  const removeFilter = (id: string) => {
    setFilters((prev) => prev.filter((row) => row.id !== id));
    if (filterPanel?.id === id) setFilterPanel(null);
  };

  const addSort = () => {
    if (!selectedTables.length) return;
    const table = selectedTables[0];
    const firstColumn = schemaMap[table]?.columns[0]?.name;
    if (!firstColumn) return;
    setSorts((prev) => [...prev, { id: uuid(), table, column: firstColumn, direction: "asc" }]);
  };

  const updateSort = (id: string, patch: Partial<SortRow>) => {
    setSorts((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  };

  const removeSort = (id: string) => {
    setSorts((prev) => prev.filter((row) => row.id !== id));
  };

  const openFilterPanel = async (row: FilterRow) => {
    if (!row.table || !row.column) return;
    const key = `${row.table}::${row.column}`;
    if (!filterOptions[key]) {
      try {
        const response = await fetchColumnValues(row.table, row.column);
        setFilterOptions((prev) => ({ ...prev, [key]: response.values }));
      } catch {
        setFilterOptions((prev) => ({ ...prev, [key]: [] }));
      }
    }
    setFilterPanel({
      id: row.id,
      table: row.table,
      column: row.column,
      search: "",
      visible: LOAD_CHUNK,
      pending: row.selectedValues || [],
    });
  };

  const applyFilterPanel = () => {
    if (!filterPanel) return;
    updateFilter(filterPanel.id, { selectedValues: filterPanel.pending });
    setFilterPanel(null);
  };

  const toggleFilterValue = (value: string) => {
    if (!filterPanel) return;
    setFilterPanel((prev) =>
      prev
        ? {
            ...prev,
            pending: prev.pending.includes(value)
              ? prev.pending.filter((item) => item !== value)
              : [...prev.pending, value],
          }
        : prev,
    );
  };

  const selectAllFilterValues = () => {
    if (!filterPanel) return;
    const key = `${filterPanel.table}::${filterPanel.column}`;
    const allValues = filterOptions[key] || [];
    setFilterPanel((prev) => (prev ? { ...prev, pending: allValues } : prev));
  };

  const clearFilterValues = () => {
    if (!filterPanel) return;
    setFilterPanel((prev) => (prev ? { ...prev, pending: [] } : prev));
  };

  const handleFilterScroll = (event: React.UIEvent<HTMLDivElement>) => {
    if (!filterPanel) return;
    const key = `${filterPanel.table}::${filterPanel.column}`;
    const values = filterOptions[key] || [];
    if (filterPanel.visible >= values.length) return;
    const target = event.currentTarget;
    if (target.scrollTop + target.clientHeight >= target.scrollHeight - 8) {
      setFilterPanel((prev) => (prev ? { ...prev, visible: Math.min(values.length, prev.visible + LOAD_CHUNK) } : prev));
    }
  };

  const filteredFilterValues = () => {
    if (!filterPanel) return [];
    const key = `${filterPanel.table}::${filterPanel.column}`;
    const values = filterOptions[key] || [];
    const filtered = filterPanel.search
      ? values.filter((value) => value.toLowerCase().includes(filterPanel.search.toLowerCase()))
      : values;
    return filtered.slice(0, filterPanel.visible);
  };

  const renderFilterPanel = () => {
    if (!filterPanel) return null;
    const key = `${filterPanel.table}::${filterPanel.column}`;
    const values = filterOptions[key] || [];
    const visibleValues = filteredFilterValues();
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-6">
        <div className="w-full max-w-sm rounded-3xl bg-white border border-slate-200 shadow-2xl p-5 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold text-slate-800">Filter values</h3>
              <p className="text-xs text-slate-500">
                {filterPanel.table}.{filterPanel.column}
              </p>
            </div>
            <button onClick={() => setFilterPanel(null)} className="text-slate-400 hover:text-slate-600 text-lg">
              ✕
            </button>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <input
              type="text"
              value={filterPanel.search}
              onChange={(event) => setFilterPanel((prev) => (prev ? { ...prev, search: event.target.value, visible: LOAD_CHUNK } : prev))}
              className="flex-1 rounded-xl border border-slate-200 px-3 py-2 text-sm"
              placeholder="Search values"
            />
            <button onClick={selectAllFilterValues} className="text-primary-600 font-semibold">
              Select all
            </button>
            <button onClick={clearFilterValues} className="text-slate-500 hover:text-slate-700">
              Clear
            </button>
          </div>
          <div
            className="max-h-60 overflow-auto rounded-2xl border border-slate-100 bg-slate-50/70 px-3 py-2 space-y-1"
            onScroll={handleFilterScroll}
          >
            {visibleValues.map((value) => (
              <label key={value} className="flex items-center gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  className="accent-primary-500"
                  checked={filterPanel.pending.includes(value)}
                  onChange={() => toggleFilterValue(value)}
                />
                <span className="truncate">{value}</span>
              </label>
            ))}
            {!values.length && <p className="text-xs text-slate-400">Loading values…</p>}
            {values.length > 0 && visibleValues.length === 0 && (
              <p className="text-xs text-slate-400">No values match your search.</p>
            )}
          </div>
          <div className="flex justify-end gap-3">
            <button
              onClick={() => setFilterPanel(null)}
              className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
            >
              Close
            </button>
            <button
              onClick={applyFilterPanel}
              className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600"
            >
              Apply
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderTablesPanel = () => {
    if (loadingSchema) return <p className="text-xs text-slate-400 mt-3">Loading schema…</p>;
    if (schemaError) return <p className="text-xs text-red-500 mt-3">{schemaError}</p>;
    return (
      <div className="mt-3 space-y-3">
        {schema.map((table) => {
          const selected = selectedTables.includes(table.name);
          const expanded = expandedTables[table.name];
          const tableLabel = TABLE_DISPLAY_NAMES[table.name] ?? table.name.replace(/_/g, " ");
          return (
            <div
              key={table.name}
              className="rounded-2xl border border-slate-200 px-3 py-2 shadow-sm bg-white/90"
            >
              <div className="flex items-center justify-between gap-3">
                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    className="mt-1 accent-primary-500"
                    checked={selected}
                    onChange={() => toggleTable(table.name)}
                  />
                  <div>
                    <div className="font-semibold text-slate-700">{tableLabel}</div>
                    <div className="text-xs text-slate-400">{table.columns.length} columns</div>
                  </div>
                </label>
                <button
                  type="button"
                  onClick={() => toggleTableExpansion(table.name)}
                  className={`text-xs font-semibold px-2 py-1 rounded-full border transition ${
                    expanded ? "border-primary-200 text-primary-600" : "border-slate-200 text-slate-500"
                  }`}
                >
                  {expanded ? "Hide columns" : "View columns"}
                </button>
              </div>
              {expanded && (
                <div className="mt-3 max-h-48 overflow-y-auto space-y-1 pr-1">
                  {table.columns.map((column) => (
                    <label key={column.name} className="flex items-center gap-2 text-sm text-slate-600">
                      <input
                        type="checkbox"
                        className="accent-primary-500"
                        checked={(selectedColumns[table.name] || []).includes(column.name)}
                        onChange={() => toggleColumn(table.name, column.name)}
                      />
                      <span>{column.name}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full overflow-auto">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
          >
            ← Back
          </button>
          <button
            onClick={handleClearAll}
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
          >
            Clear All
          </button>
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-semibold text-slate-800">Data Export</h1>
          <p className="text-sm text-slate-500">
            Pick tables, columns and filters - the preview refreshes automatically as you build your export.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={triggerDownload}
            disabled={!hasColumnSelection || downloading}
            className="rounded-full border border-primary-200 px-4 py-2 text-sm font-semibold text-primary-600 hover:bg-primary-50 disabled:opacity-50"
          >
            {downloading ? "Preparing…" : "Download CSV"}
          </button>
          <button
            onClick={openEmailModal}
            disabled={!hasColumnSelection}
            className="rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100 disabled:opacity-50"
          >
            Email CSV
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.6fr_2.4fr] gap-5 flex-1 min-h-0 auto-rows-[minmax(0,1fr)] overflow-hidden">
        <div className="xl:col-span-1 rounded-3xl border border-primary-100 bg-white/95 shadow-lg p-5 flex flex-col min-h-[280px] max-h-full">
          <div className="pb-2">
            <h2 className="text-sm font-semibold text-primary-600 uppercase tracking-wide">Tables</h2>
            <p className="text-xs text-slate-400">Expand any table to browse its columns and pick exactly what you need.</p>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto pr-2">{renderTablesPanel()}</div>
        </div>

        <div className="rounded-3xl border border-primary-100 bg-white/95 shadow-lg p-5 flex flex-col min-h-[320px] max-h-full overflow-hidden">
          <div className="rounded-3xl border border-slate-100 bg-white/90 p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-700">Row limit</h3>
                <p className="text-xs text-slate-400">Preview capped at 500 rows.</p>
              </div>
              <input
                type="number"
                min={10}
                max={500}
                value={limit}
                onChange={(event) =>
                  setLimit(Math.max(10, Math.min(500, Number(event.target.value) || DEFAULT_LIMIT)))
                }
                className="w-32 rounded-xl border border-slate-200 px-3 py-1.5 text-sm"
              />
            </div>
          </div>
          <div className="flex flex-col gap-4 flex-1 min-h-0 overflow-y-auto pr-1">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0 overflow-x-hidden">
              <div className="rounded-3xl border border-slate-100 bg-white/90 p-4 flex flex-col min-h-0 max-h-[360px] lg:max-h-none">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-slate-700">Filters</h3>
                  <button
                    onClick={addFilter}
                    className="text-xs font-semibold text-primary-600 hover:text-primary-700 disabled:text-slate-300"
                    disabled={!selectedTables.length}
                  >
                    + Add filter
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto pr-1 space-y-2.5">
                  {filters.length === 0 ? (
                    <p className="text-xs text-slate-400">No filters applied.</p>
                  ) : (
                    filters.map((row) => (
                      <div key={row.id} className="rounded-2xl border border-slate-200 px-2.5 py-2.5 space-y-2.5 min-w-0">
                        <div className="flex items-center gap-2 min-w-0">
                          <select
                            value={row.table}
                            onChange={(event) =>
                              updateFilter(row.id, { table: event.target.value, selectedValues: [] })
                            }
                            className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-sm min-w-0"
                          >
                            {selectedTables.map((table) => (
                              <option key={table} value={table}>
                                {table}
                              </option>
                            ))}
                          </select>
                          <button onClick={() => removeFilter(row.id)} className="text-slate-400 hover:text-red-500 text-sm">
                            ✕
                          </button>
                        </div>
                        <div className="flex items-center gap-1.5 min-w-0">
                          <select
                            value={row.column}
                            onChange={(event) =>
                              updateFilter(row.id, { column: event.target.value, selectedValues: [] })
                            }
                            className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-sm min-w-0"
                          >
                            {(schemaMap[row.table]?.columns || []).map((column) => (
                              <option key={column.name} value={column.name}>
                                {column.name}
                              </option>
                            ))}
                          </select>
                          <select
                            value={row.operator}
                            onChange={(event) => updateFilter(row.id, { operator: event.target.value })}
                            className="rounded-xl border border-slate-200 px-2 py-1 text-sm"
                          >
                            {OPERATORS.map((operator) => (
                              <option key={operator.value} value={operator.value}>
                                {operator.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        {row.operator === "between" ? (
                          <div className="space-y-1.5">
                            <input
                              type="text"
                              value={row.value}
                              onChange={(event) => updateFilter(row.id, { value: event.target.value })}
                              className="w-full rounded-xl border border-slate-200 px-3 py-1 text-sm"
                              placeholder="From"
                            />
                            <input
                              type="text"
                              value={row.valueB ?? ""}
                              onChange={(event) => updateFilter(row.id, { valueB: event.target.value })}
                              className="w-full rounded-xl border border-slate-200 px-3 py-1 text-sm"
                              placeholder="To"
                            />
                          </div>
                        ) : (
                          <input
                            type="text"
                            value={row.value}
                            onChange={(event) => updateFilter(row.id, { value: event.target.value })}
                            className="w-full rounded-xl border border-slate-200 px-3 py-1 text-sm"
                            placeholder="Value"
                            disabled={Boolean(row.selectedValues?.length)}
                          />
                        )}
                        <div className="rounded-2xl border border-slate-200 bg-slate-50/60 px-2.5 py-1.5">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-slate-500">
                              {row.selectedValues?.length ? `${row.selectedValues.length} values selected` : "No values selected"}
                            </span>
                            <button
                              onClick={() => openFilterPanel(row)}
                              className="text-xs font-semibold text-primary-600 hover:text-primary-700"
                            >
                              Choose values
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div className="rounded-3xl border border-slate-100 bg-white/90 p-4 flex flex-col min-h-0 max-h-[360px] lg:max-h-none">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-slate-700">Sorting</h3>
                  <button
                    onClick={addSort}
                    className="text-xs font-semibold text-primary-600 hover:text-primary-700 disabled:text-slate-300"
                    disabled={!selectedTables.length}
                  >
                    + Add sort
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto pr-1 space-y-2.5">
                  {sorts.length === 0 ? (
                    <p className="text-xs text-slate-400">No sorting applied.</p>
                  ) : (
                    sorts.map((row) => (
                      <div key={row.id} className="rounded-2xl border border-slate-200 px-2.5 py-2.5 space-y-1.5 min-w-0">
                        <div className="flex items-center gap-2 min-w-0">
                          <select
                            value={row.table}
                            onChange={(event) => updateSort(row.id, { table: event.target.value })}
                            className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-sm"
                          >
                            {selectedTables.map((table) => (
                              <option key={table} value={table}>
                                {table}
                              </option>
                            ))}
                          </select>
                          <button onClick={() => removeSort(row.id)} className="text-slate-400 hover:text-red-500 text-sm">
                            ✕
                          </button>
                        </div>
                        <div className="flex items-center gap-1.5 min-w-0">
                          <select
                            value={row.column}
                            onChange={(event) => updateSort(row.id, { column: event.target.value })}
                            className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-sm"
                          >
                            {(schemaMap[row.table]?.columns || []).map((column) => (
                              <option key={column.name} value={column.name}>
                                {column.name}
                              </option>
                            ))}
                          </select>
                          <select
                            value={row.direction}
                            onChange={(event) => updateSort(row.id, { direction: event.target.value as "asc" | "desc" })}
                            className="rounded-xl border border-slate-200 px-2 py-1 text-sm"
                          >
                            <option value="asc">Ascending</option>
                            <option value="desc">Descending</option>
                          </select>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
            <div className="rounded-3xl border border-slate-100 bg-white/90 p-4 flex flex-col flex-1 min-h-[280px]">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-700">Preview</h3>
                <div className="flex items-center gap-2 text-xs">
                  {autoRefreshing && <span className="text-slate-400">Updating…</span>}
                  {preview?.sql && (
                    <button
                      onClick={() => setSqlModalOpen(true)}
                      className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-primary-600 hover:bg-primary-50"
                    >
                      View SQL
                    </button>
                  )}
                </div>
              </div>
              {!preview ? (
                <p className="text-sm text-slate-400">Select tables and columns to see a preview.</p>
              ) : (
                <div className="flex-1 min-h-0 overflow-x-auto overflow-y-auto border border-slate-200 rounded-2xl pb-3">
                  <div className="w-full" style={{ minWidth: previewMinWidth }}>
                    <table className="divide-y divide-slate-200 text-sm">
                      <thead className="bg-slate-100/60 text-slate-600">
                        <tr>
                          {preview.columns.map((column) => (
                            <th key={column} className="px-3 py-2 text-left font-semibold">
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {preview.rows.map((row, rowIdx) => (
                          <tr key={rowIdx}>
                            {preview.columns.map((column) => (
                              <td key={column} className="px-3 py-2 text-slate-600 whitespace-pre">
                                {String(row[column] ?? "")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {emailModalOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/40 px-6">
          <div className="w-full max-w-md rounded-3xl bg-white border border-slate-200 shadow-2xl p-6 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-800">Email CSV export</h3>
                <p className="text-sm text-slate-500">Send this selection to any email address.</p>
              </div>
              <button onClick={() => setEmailModalOpen(false)} className="text-slate-400 hover:text-slate-600 text-lg">
                ✕
              </button>
            </div>
            <label className="text-sm font-medium text-slate-600">
              Recipient email
              <input
                type="email"
                value={emailAddress}
                onChange={(event) => setEmailAddress(event.target.value)}
                className="mt-2 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
                placeholder="name@example.com"
              />
            </label>
            {emailStatus && <p className="text-sm text-emerald-600">{emailStatus}</p>}
            {emailError && <p className="text-sm text-red-500">{emailError}</p>}
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setEmailModalOpen(false)}
                className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                onClick={handleEmailSubmit}
                disabled={!emailAddress.trim() || emailSending}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600 disabled:opacity-60"
              >
                {emailSending ? "Sending…" : "Send"}
              </button>
            </div>
          </div>
        </div>
      )}

      {sqlModalOpen && preview?.sql && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/40 px-6">
          <div className="w-full max-w-2xl rounded-3xl bg-white border border-slate-200 shadow-2xl p-6 space-y-4">
            <div className="flex items-start justify-between">
              <h3 className="text-lg font-semibold text-slate-800">Generated SQL</h3>
              <button onClick={() => setSqlModalOpen(false)} className="text-slate-400 hover:text-slate-600 text-lg">
                ✕
              </button>
            </div>
            <div className="max-h-[60vh] overflow-auto rounded-2xl bg-slate-950 text-slate-100 p-4 text-xs">
              <pre>
                <code>{preview.sql}</code>
              </pre>
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setSqlModalOpen(false)}
                className="rounded-full bg-primary-500 text-white px-5 py-2 text-sm font-semibold shadow-md hover:bg-primary-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {renderFilterPanel()}
    </div>
  );
};