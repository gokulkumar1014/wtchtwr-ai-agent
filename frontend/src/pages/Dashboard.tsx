import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  Bar,
  Legend,
  Label,
} from "recharts";
import { useNavigate } from "react-router-dom";
import DeckGL from "@deck.gl/react";
import { MapView, type ViewState, type Tooltip } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, ScatterplotLayer } from "@deck.gl/layers";
import {
  DashboardComparisonPayload,
  DashboardFilterOptions,
  DashboardFiltersPayload,
  DashboardMapListing,
  DashboardViewResponse,
  fetchDashboardFilters,
  fetchDashboardView,
} from "@/lib/api";

const GROUP_COLORS: Record<string, string> = {
  Highbury: "#4338ca",
  Comparison: "#0284c7",
};

const GROUP_STROKE_COLORS: Record<string, string> = {
  Highbury: "#1e1b4b",
  Comparison: "rgba(14, 116, 144, 0.55)",
};

const ROOM_TYPE_COLOR_PALETTE = [
  "#2563eb",
  "#ef4444",
  "#f59e0b",
  "#10b981",
  "#a855f7",
  "#14b8a6",
  "#f97316",
  "#ec4899",
];
const DEFAULT_ROOM_TYPE_COLOR = "#64748b";

const TOOLTIP_BG = "rgba(15, 23, 42, 0.95)";   // dark slate background @95%
const TOOLTIP_TEXT = "#F9FAFB";                // bright text (slate-50)
const TOOLTIP_MUTED = "#CBD5E1";               // muted secondary text (slate-300)
const TOOLTIP_BORDER = "rgba(148, 163, 184, 0.28)"; // subtle border (slate-400 @28%)
const TOOLTIP_SHADOW = "0 10px 28px rgba(2, 6, 23, 0.35)"; // soft shadow for depth

const KNOWN_ROOM_TYPE_COLORS: Record<string, string> = {
  "Entire home/apt": "#2563eb",
  "Private room": "#ef4444",
  "Hotel room": "#10b981",
  "Shared room": "#f59e0b",
};

const generatedRoomTypeColors: Record<string, string> = {};

type MapViewMode = "both" | "highbury" | "comparison";
type ExpandedViz = "map" | "guestExperience" | "ratingDistribution";

const INITIAL_VIEW_STATE: ViewState = {
  longitude: -73.97,
  latitude: 40.72,
  zoom: 10.6,
  pitch: 0,
  bearing: 0,
};

const niceStep = (raw: number): number => {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const exponent = Math.floor(Math.log10(raw));
  const fraction = raw / Math.pow(10, exponent);
  let niceFraction: number;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  else niceFraction = 10;
  return niceFraction * Math.pow(10, exponent);
};

const createNiceTicks = (maxValue: number, approxSteps = 4): number[] => {
  const max = Math.max(0, Math.ceil(maxValue));
  if (max === 0) return [0, 1];
  const roughStep = Math.ceil(max / Math.max(1, approxSteps));
  const step = niceStep(roughStep);
  const ticks: number[] = [0];
  for (let value = step; value < max; value += step) ticks.push(value);
  if (ticks[ticks.length - 1] !== max) ticks.push(max);
  return ticks;
};

const hexToRgb = (hex: string): [number, number, number] => {
  const normalized = hex.replace("#", "");
  const bigint = parseInt(normalized, 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
};

const DEFAULT_FILTERS: DashboardFiltersPayload = {
  neighborhoodGroups: [],
  neighborhoods: [],
  propertyTypes: [],
  roomTypes: [],
  accommodates: {},
  bathrooms: {},
  bedrooms: {},
  beds: {},
  price: {},
  hostNames: [], // kept in payload for compatibility, but no UI control for this anymore
  bathroomDetails: [],
};

const DEFAULT_COMPARISON: DashboardComparisonPayload = {
  mode: "market",
  hosts: [],
};

const formatNumber = (value: number | null, options?: Intl.NumberFormatOptions) => {
  if (value === null || Number.isNaN(value)) return "–";
  return new Intl.NumberFormat("en-US", options).format(value);
};

const formatPercentage = (value: number | null, digits = 1) => {
  if (value === null || Number.isNaN(value)) return "–";
  return `${value.toFixed(digits)}%`;
};

const resolveRoomTypeColor = (roomType: string | null): string => {
  if (!roomType) return DEFAULT_ROOM_TYPE_COLOR;
  if (KNOWN_ROOM_TYPE_COLORS[roomType]) return KNOWN_ROOM_TYPE_COLORS[roomType];
  if (generatedRoomTypeColors[roomType]) return generatedRoomTypeColors[roomType];
  const hash = Math.abs(Array.from(roomType).reduce((acc, ch) => acc + ch.charCodeAt(0), 0));
  const color = ROOM_TYPE_COLOR_PALETTE[hash % ROOM_TYPE_COLOR_PALETTE.length] ?? DEFAULT_ROOM_TYPE_COLOR;
  generatedRoomTypeColors[roomType] = color;
  return color;
};

const DashboardSkeleton: React.FC = () => (
  <div className="flex-1 flex items-center justify-center">
    <div className="animate-pulse text-sm text-slate-400">Loading dashboard insights…</div>
  </div>
);

function listingColor(listing: DashboardMapListing): string {
  return resolveRoomTypeColor(listing.roomType ?? null);
}

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [filterOptions, setFilterOptions] = useState<DashboardFilterOptions | null>(null);
  const [filters, setFilters] = useState<DashboardFiltersPayload>(DEFAULT_FILTERS);
  const [comparison, setComparison] = useState<DashboardComparisonPayload>(DEFAULT_COMPARISON);
  const [insights, setInsights] = useState<DashboardViewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filtersInitialised, setFiltersInitialised] = useState(false);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [mapViewMode, setMapViewMode] = useState<MapViewMode>("both");
  const [mapViewState, setMapViewState] = useState<ViewState>(INITIAL_VIEW_STATE);
  const [expandedViz, setExpandedViz] = useState<ExpandedViz | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const startedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
        const options = await fetchDashboardFilters();
        if (!active) return;
        setFilterOptions(options);
        setFilters((prev) => ({
          ...prev,
          accommodates: { min: options.ranges.accommodates.min, max: options.ranges.accommodates.max },
          bathrooms: { min: options.ranges.bathrooms.min, max: options.ranges.bathrooms.max },
          bedrooms: { min: options.ranges.bedrooms.min, max: options.ranges.bedrooms.max },
          beds: { min: options.ranges.beds.min, max: options.ranges.beds.max },
          price: { min: options.ranges.price.min, max: Math.min(options.ranges.price.max ?? 2500, 2500) },
        }));
        setFiltersInitialised(true);
        const endedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
        console.info(`[Dashboard] Loaded filter metadata in ${Math.round(endedAt - startedAt)}ms`);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load dashboard filters.");
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    const startedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
    try {
      const payload = { filters, comparison };
      const data = await fetchDashboardView(payload);
      setInsights(data);
      setMapViewMode((m) => m);
      const endedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
      console.info(`[Dashboard] Loaded insights in ${Math.round(endedAt - startedAt)}ms`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load dashboard insights.");
    } finally {
      setLoading(false);
    }
  }, [filters, comparison]);

  useEffect(() => {
    if (filtersInitialised && !hasLoadedOnce) {
      loadDashboard().finally(() => setHasLoadedOnce(true));
    }
  }, [filtersInitialised, hasLoadedOnce, loadDashboard]);

  useEffect(() => {
    if (!filtersInitialised) return;
    const h = window.setTimeout(() => loadDashboard(), 300);
    return () => window.clearTimeout(h);
  }, [filters, comparison, filtersInitialised, loadDashboard]);

  const handleResetFilters = () => {
    if (!filterOptions) return;
    setFilters({
      neighborhoodGroups: [],
      neighborhoods: [],
      propertyTypes: [],
      roomTypes: [],
      accommodates: { min: filterOptions.ranges.accommodates.min, max: filterOptions.ranges.accommodates.max },
      bathrooms: { min: filterOptions.ranges.bathrooms.min, max: filterOptions.ranges.bathrooms.max },
      bedrooms: { min: filterOptions.ranges.bedrooms.min, max: filterOptions.ranges.bedrooms.max },
      beds: { min: filterOptions.ranges.beds.min, max: filterOptions.ranges.beds.max },
      price: { min: filterOptions.ranges.price.min, max: Math.min(filterOptions.ranges.price.max ?? 2500, 2500) },
      hostNames: [], // still sent (no UI control)
      bathroomDetails: [],
    });
    setComparison(DEFAULT_COMPARISON);
  };

  const toggleValue = useCallback((values: string[], value: string) => {
    if (values.includes(value)) return values.filter((v) => v !== value);
    return [...values, value];
  }, []);

  const comparisonLabel = comparison.mode === "market" ? "Market" : "Selected hosts";

  const occupancyCards = insights?.summary.occupancyCards ?? [];
  const revenueCards = insights?.summary.revenueCards ?? [];

  const reviewScoreChartData = useMemo(() => {
    if (!insights) return [];
    return insights.summary.reviewScores.map((e) => ({
      label: e.label,
      highbury: e.highbury ?? 0,
      comparison: e.comparison ?? 0,
    }));
  }, [insights]);

  const ratingDistributionData = useMemo(() => {
    if (!insights) return [];
    return insights.summary.ratingSummary.distribution.map((e) => ({
      label: e.label,
      highbury: e.highbury,
      comparison: e.comparison,
    }));
  }, [insights]);

  const maxHighburyCount = useMemo(
    () => (ratingDistributionData.length ? Math.max(...ratingDistributionData.map((e) => e.highbury)) : 0),
    [ratingDistributionData]
  );
  const maxComparisonCount = useMemo(
    () => (ratingDistributionData.length ? Math.max(...ratingDistributionData.map((e) => e.comparison)) : 0),
    [ratingDistributionData]
  );
  const highburyTicks = useMemo(() => createNiceTicks(maxHighburyCount, 4), [maxHighburyCount]);
  const comparisonTicks = useMemo(() => createNiceTicks(maxComparisonCount, 5), [maxComparisonCount]);
  const highburyAxisMax = highburyTicks[highburyTicks.length - 1] ?? 1;
  const comparisonAxisMax = comparisonTicks[comparisonTicks.length - 1] ?? 1;

  const mapListings = useMemo(() => {
    if (!insights) return [];
    return insights.map.listings.filter((l) =>
      mapViewMode === "both" ? true : mapViewMode === "highbury" ? l.group === "Highbury" : l.group !== "Highbury"
    );
  }, [insights, mapViewMode]);

  const roomTypeLegend = useMemo(() => {
    if (!mapListings.length) return [];
    const seen = new Map<string, string>();
    mapListings.forEach((l) => {
      const key = l.roomType ?? "Unknown";
      if (!seen.has(key)) seen.set(key, resolveRoomTypeColor(l.roomType ?? null));
    });
    return Array.from(seen.entries()).map(([roomType, color]) => ({ roomType, color }));
  }, [mapListings]);

  const totals = insights?.summary.totals;
  const ratingSummary = insights?.summary.ratingSummary;
  const priceSummary = insights?.summary.priceSummary;
  const combinedHostCounts = insights?.summary.hostCounts.combined ?? [];

  const PRICE_KEYS = ["min", "q1", "median", "q3", "max"] as const;

  const neighborhoodsOptions = filterOptions?.neighborhoods ?? [];
  const propertyTypeOptions = filterOptions?.propertyTypes ?? [];
  const roomTypeOptions = filterOptions?.roomTypes ?? [];
  const hostNameOptions = filterOptions?.hostNames ?? [];
  const bathroomDetailOptions = filterOptions?.bathroomDetails ?? [];

  const hostNameOptionsSorted = useMemo(() => {
    const options = [...hostNameOptions];
    options.sort((a, b) => {
      const aH = a.toLowerCase() === "highbury";
      const bH = b.toLowerCase() === "highbury";
      if (aH && !bH) return -1;
      if (!aH && bH) return 1;
      return a.localeCompare(b);
    });
    return options;
  }, [hostNameOptions]);

  const deckMapLayers = useMemo(() => {
    // Switched to a highly reliable OSM tile endpoint
    const baseLayer = new TileLayer({
      id: "base-map",
      data: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props) => {
        const { west, south, east, north } = props.tile.bbox;
        return new BitmapLayer(props, { data: null, image: props.data, bounds: [west, south, east, north] });
      },
    });

    if (mapListings.length === 0) return [baseLayer];

    const scatterLayer = new ScatterplotLayer<DashboardMapListing>({
      id: "listings",
      data: mapListings,
      radiusUnits: "pixels",
      radiusMinPixels: 1.5,
      radiusMaxPixels: 5,
      getPosition: (d) => [d.lng, d.lat],
      getRadius: (d) => (d.group === "Highbury" ? 3.2 : 2.2),
      stroked: true,
      lineWidthUnits: "pixels",
      lineWidthMinPixels: 0.6,
      lineWidthMaxPixels: 1.1,
      getLineColor: (d) => (d.group === "Highbury" ? [59, 35, 124, 160] : [30, 87, 128, 140]),
      getFillColor: (d) => {
        const [r, g, b] = hexToRgb(listingColor(d));
        const alpha = d.group === "Highbury" ? 165 : 135;
        return [r, g, b, alpha];
      },
      pickable: true,
      parameters: { depthTest: false },
      updateTriggers: { getFillColor: mapListings, getLineColor: mapListings },
    });

    return [baseLayer, scatterLayer];
  }, [mapListings]);

  const deckTooltip = useCallback(
    ({ object }: { object?: DashboardMapListing | null }): Tooltip | null => {
      if (!object) return null;

      const location =
        [object.neighborhood, object.neighborhoodGroup].filter(Boolean).join(" · ") || "New York City";
      const property =
        [object.propertyType, object.roomType].filter(Boolean).join(" · ") || "Listing";

      return {
        html: `
          <div style="font-size:12px; line-height:1.25;">
            <div style="font-weight:700; color:${TOOLTIP_TEXT}; margin-bottom:4px;">
              ${object.propertyName ?? object.hostName ?? "Listing"}
            </div>
            <div style="color:${TOOLTIP_MUTED}; margin-bottom:2px;">${location}</div>
            <div style="color:${TOOLTIP_MUTED}; margin-bottom:6px;">${property}</div>
            <div style="color:${TOOLTIP_TEXT};">Host: <span style="color:${TOOLTIP_MUTED};">${object.hostName ?? "—"}</span></div>
            <div style="color:${TOOLTIP_TEXT};">Price: <span style="color:${TOOLTIP_MUTED};">
              ${formatNumber(object.price, { style: "currency", currency: "USD", maximumFractionDigits: 0 })}
            </span></div>
            <div style="color:${TOOLTIP_TEXT};">Overall rating: <span style="color:${TOOLTIP_MUTED};">
              ${formatNumber(object.reviewScore, { maximumFractionDigits: 2 })}
            </span></div>
          </div>
        `,
        style: {
          backgroundColor: TOOLTIP_BG,
          color: TOOLTIP_TEXT,
          padding: "10px 12px",
          borderRadius: "12px",
          border: `1px solid ${TOOLTIP_BORDER}`,
          boxShadow: TOOLTIP_SHADOW,
          backdropFilter: "blur(2px)",
          maxWidth: "300px",
        },
      };
    },
    []
  );

  const handleZoom = useCallback((delta: number) => {
    setMapViewState((prev) => {
      const nextZoom = Math.min(Math.max(prev.zoom + delta, 3), 16);
      return { ...prev, zoom: nextZoom };
    });
  }, []);

  const renderMapVisualization = (height: number) => (
    <div
      className="rounded-3xl overflow-hidden border border-slate-200"
      style={{ height, minHeight: 320, position: "relative", width: "100%" }}
    >
      <DeckGL
        controller={{ dragRotate: false }}
        views={new MapView({ repeat: true })}
        viewState={mapViewState}
        onViewStateChange={({ viewState }) => setMapViewState(viewState as ViewState)}
        layers={deckMapLayers}
        glOptions={{ antialias: true }}
        getTooltip={deckTooltip}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />
      <div className="absolute top-3 right-3 flex flex-col gap-2">
        <button
          onClick={() => handleZoom(0.6)}
          className="h-9 w-9 rounded-full border border-white/60 bg-white/80 font-semibold text-slate-700 shadow-sm hover:bg-white transition-colors"
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          onClick={() => handleZoom(-0.6)}
          className="h-9 w-9 rounded-full border border-white/60 bg-white/80 font-semibold text-slate-700 shadow-sm hover:bg-white transition-colors"
          aria-label="Zoom out"
        >
          −
        </button>
      </div>
    </div>
  );

  const renderMapLegend = () => (
    <div className="flex flex-wrap items-center gap-3 mt-4 text-xs text-slate-500">
      {roomTypeLegend.map(({ roomType, color }) => (
        <span key={roomType} className="inline-flex items-center gap-2 bg-slate-100/60 rounded-full px-3 py-1">
          <span className="inline-flex h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
          <span className="capitalize text-slate-600">{roomType.toLowerCase()}</span>
        </span>
      ))}
      <span className="inline-flex items-center gap-2 bg-slate-100/60 rounded-full px-3 py-1">
        <span className="inline-flex h-3 w-6 rounded-full border-2" style={{ borderColor: GROUP_STROKE_COLORS.Highbury }} />
        <span className="text-slate-600 font-semibold">Highbury outline</span>
      </span>
      <span className="inline-flex items-center gap-2 bg-slate-100/60 rounded-full px-3 py-1">
        <span className="inline-flex h-3 w-6 rounded-full border" style={{ borderColor: GROUP_STROKE_COLORS.Comparison }} />
        <span className="text-slate-600">{comparisonLabel} outline</span>
      </span>
      <span>
        {formatNumber(totals?.highburyListings ?? 0, { maximumFractionDigits: 0 })} Highbury ·{" "}
        {formatNumber(totals?.comparisonListings ?? 0, { maximumFractionDigits: 0 })} {comparisonLabel.toLowerCase()}
      </span>
    </div>
  );

  const GuestExperienceChart: React.FC<{ height: number }> = ({ height }) => (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={reviewScoreChartData} layout="vertical" margin={{ top: 24, right: 32, left: 8, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
          <XAxis type="number" domain={[0, 5]} tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="label" tick={{ fontSize: 12, width: 120 }} width={130} tickLine={false} />
          <RechartsTooltip formatter={(v: number) => formatNumber(v, { maximumFractionDigits: 2 })} />
          <Legend />
          <Bar dataKey="highbury" fill={GROUP_COLORS.Highbury} name="Highbury" radius={[0, 6, 6, 0]} barSize={14} />
          <Bar dataKey="comparison" fill={GROUP_COLORS.Comparison} name={comparisonLabel} radius={[0, 6, 6, 0]} barSize={14} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  const RatingDistributionChart: React.FC<{ height: number }> = ({ height }) => (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={ratingDistributionData} margin={{ top: 28, right: 32, left: 8, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 12 }} />
          <YAxis
            yAxisId="left"
            allowDecimals={false}
            tick={{ fontSize: 11 }}
            domain={[0, highburyAxisMax]}
            ticks={highburyTicks}
            tickFormatter={(v: number) => formatNumber(v, { maximumFractionDigits: 0 })}
          >
            <Label value="Highbury listings" angle={-90} position="insideLeft" style={{ fill: GROUP_COLORS.Highbury, fontSize: 11 }} offset={-5} />
          </YAxis>
          <YAxis
            yAxisId="right"
            orientation="right"
            allowDecimals={false}
            tick={{ fontSize: 11 }}
            domain={[0, comparisonAxisMax]}
            ticks={comparisonTicks}
            tickFormatter={(v: number) => formatNumber(v, { maximumFractionDigits: 0 })}
          >
            <Label
              value={`${comparisonLabel} listings`}
              angle={90}
              position="insideRight"
              style={{ fill: GROUP_COLORS.Comparison, fontSize: 11 }}
              offset={-5}
            />
          </YAxis>
          <RechartsTooltip formatter={(v: number) => formatNumber(v, { maximumFractionDigits: 0 })} />
          <Legend />
          <Bar yAxisId="left" dataKey="highbury" fill={GROUP_COLORS.Highbury} name="Highbury listings" radius={[6, 6, 0, 0]} />
          <Bar yAxisId="right" dataKey="comparison" fill={GROUP_COLORS.Comparison} name={`${comparisonLabel} listings`} radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="h-full rounded-3xl border border-primary-100 bg-white shadow-2xl overflow-hidden flex flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-primary-100 px[6] py[4] px-6 py-4 bg-white/95">
        <button
          onClick={() => navigate(-1)}
          className="inline-flex items-center gap-2 rounded-full border border-primary-200 px-4 py-2 text-sm font-semibold text-primary-600 hover:bg-primary-50 transition-colors"
        >
          ← Back
        </button>
        <div className="text-center">
          <div className="text-lg font-semibold text-slate-800">wtchtwr - NYC Portfolio Insights</div>
          <div className="text-xs text-slate-500">Compare Highbury performance against the broader market or specific competitors.</div>
        </div>
        <div />
      </div>

      <div className="flex-1 overflow-hidden flex flex-col xl:flex-row">
        {/* FILTER PANEL */}
        <aside className="xl:w-80 border-b xl:border-b-0 xl:border-r border-slate-200/60 bg-white/90 px-5 py-6 space-y-6 overflow-y-auto">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-700">Filters</h2>
            <button onClick={handleResetFilters} className="text-xs font-medium text-primary-600 hover:text-primary-700">
              Reset
            </button>
          </div>

          <div className="space-y-4 text-sm text-slate-600">
            {/* Neighbourhood groups */}
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Neighbourhood groups</label>
              <div className="max-h-40 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                <div className="space-y-1.5 py-1">
                  {(filterOptions?.neighborhoodGroups ?? []).map((group) => {
                    const active = filters.neighborhoodGroups.includes(group);
                    return (
                      <label
                        key={group}
                        className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                          active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() =>
                            setFilters((prev) => ({ ...prev, neighborhoodGroups: toggleValue(prev.neighborhoodGroups, group) }))
                          }
                          className="h-4 w-4 accent-primary-500"
                        />
                        <span>{group}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Neighbourhoods */}
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Neighbourhoods</label>
              <div className="max-h-48 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                <div className="space-y-1.5 py-1">
                  {neighborhoodsOptions.map((n) => {
                    const active = filters.neighborhoods.includes(n);
                    return (
                      <label
                        key={n}
                        className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                          active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => setFilters((prev) => ({ ...prev, neighborhoods: toggleValue(prev.neighborhoods, n) }))}
                          className="h-4 w-4 accent-primary-500"
                        />
                        <span>{n}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Property types */}
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Property types</label>
              <div className="max-h-40 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                <div className="space-y-1.5 py-1">
                  {propertyTypeOptions.map((p) => {
                    const active = filters.propertyTypes.includes(p);
                    return (
                      <label
                        key={p}
                        className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                          active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => setFilters((prev) => ({ ...prev, propertyTypes: toggleValue(prev.propertyTypes, p) }))}
                          className="h-4 w-4 accent-primary-500"
                        />
                        <span>{p}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Room types */}
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Room types</label>
              <div className="max-h-32 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                <div className="space-y-1.5 py-1">
                  {roomTypeOptions.map((t) => {
                    const active = filters.roomTypes.includes(t);
                    return (
                      <label
                        key={t}
                        className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                          active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => setFilters((prev) => ({ ...prev, roomTypes: toggleValue(prev.roomTypes, t) }))}
                          className="h-4 w-4 accent-primary-500"
                        />
                        <span>{t}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Numeric ranges */}
            {([
              { label: "Accommodates", key: "accommodates" as const },
              { label: "Bathrooms", key: "bathrooms" as const },
              { label: "Bedrooms", key: "bedrooms" as const },
              { label: "Beds", key: "beds" as const },
              { label: "Price (USD)", key: "price" as const },
            ]).map((item) => {
              const rangeValue = filters[item.key];
              return (
                <div key={item.key} className="grid grid-cols-2 gap-2 items-center">
                  <label className="col-span-2 font-semibold text-slate-700">{item.label}</label>
                  <input
                    type="number"
                    value={rangeValue.min ?? ""}
                    onChange={(e) =>
                      setFilters((prev) => {
                        const nextRange = { ...prev[item.key], min: e.target.value === "" ? undefined : Number(e.target.value) };
                        return { ...prev, [item.key]: nextRange };
                      })
                    }
                    className="rounded-2xl border border-slate-200 px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-200"
                    placeholder="Min"
                  />
                  <input
                    type="number"
                    value={rangeValue.max ?? ""}
                    onChange={(e) =>
                      setFilters((prev) => {
                        const nextRange = { ...prev[item.key], max: e.target.value === "" ? undefined : Number(e.target.value) };
                        return { ...prev, [item.key]: nextRange };
                      })
                    }
                    className="rounded-2xl border border-slate-200 px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-200"
                    placeholder="Max"
                  />
                </div>
              );
            })}

            {/* Bathroom detail contains */}
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Bathroom details</label>
              <div className="max-h-48 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                <div className="space-y-1.5 py-1">
                  {bathroomDetailOptions.map((detail) => {
                    const active = filters.bathroomDetails.includes(detail);
                    return (
                      <label
                        key={detail}
                        className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                          active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => setFilters((prev) => ({ ...prev, bathroomDetails: toggleValue(prev.bathroomDetails, detail) }))}
                          className="h-4 w-4 accent-primary-500"
                        />
                        <span>{detail}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Comparison mode (includes Select hosts) */}
            <div className="border-t border-slate-200 pt-4 space-y-3">
              <div>
                <label className="font-semibold text-slate-700 block mb-1">Comparison mode</label>
                <select
                  value={comparison.mode}
                  onChange={(e) => setComparison((prev) => ({ ...prev, mode: e.target.value as "market" | "hosts" }))}
                  className="w-full rounded-2xl border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-200"
                >
                  <option value="market">Highbury vs. Market</option>
                  <option value="hosts">Highbury vs. selected hosts</option>
                </select>
              </div>
              {comparison.mode === "hosts" && (
                <div>
                  <label className="font-semibold text-slate-700 block mb-1">Select hosts</label>
                  <div className="max-h-40 overflow-y-auto rounded-2xl border border-slate-200 bg-white divide-y divide-slate-100">
                    <div className="space-y-1.5 py-1">
                      {hostNameOptionsSorted.map((host) => {
                        const active = comparison.hosts.includes(host);
                        return (
                          <label
                            key={host}
                            className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-2xl transition ${
                              active ? "bg-primary-50 text-primary-700 font-semibold" : "hover:bg-slate-50 text-slate-600"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={active}
                              onChange={() => setComparison((prev) => ({ ...prev, hosts: toggleValue(prev.hosts, host) }))}
                              className="h-4 w-4 accent-primary-500"
                            />
                            <span>{host}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {error && <p className="text-sm text-red-500">{error}</p>}
            <p className="text-xs text-slate-400">
              Choose *Highbury vs. selected hosts* to compare directly against specific competitors you select.
            </p>
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <main className="flex-1 overflow-y-auto bg-slate-50/60 p-5 sm:p-6 space-y-6">
          {!insights && (loading || !hasLoadedOnce) ? (
            <DashboardSkeleton />
          ) : insights ? (
            <>
              {/* Map card */}
              <section className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700">New York City coverage</h3>
                    <p className="text-xs text-slate-400">
                      Markers are colour-coded by room type; outline shows Highbury vs. {comparisonLabel.toLowerCase()}. Displaying{" "}
                      {formatNumber(insights.map.rendered, { maximumFractionDigits: 0 })} listings (
                      {formatNumber(totals?.highburyListings ?? 0, { maximumFractionDigits: 0 })} Highbury ·{" "}
                      {formatNumber(totals?.comparisonListings ?? 0, { maximumFractionDigits: 0 })} {comparisonLabel.toLowerCase()}
                      ).
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="inline-flex rounded-full border border-slate-200 bg-slate-50/80 p-1">
                      {[
                        { label: "Both", value: "both" as MapViewMode },
                        { label: "Highbury", value: "highbury" as MapViewMode },
                        { label: comparisonLabel, value: "comparison" as MapViewMode },
                      ].map((item) => {
                        const active = mapViewMode === item.value;
                        return (
                          <button
                            key={item.value}
                            onClick={() => setMapViewMode(item.value)}
                            className={`px-3 py-1.5 text-xs font-semibold rounded-full transition-colors ${
                              active ? "bg-white text-primary-700 shadow-sm" : "text-slate-500 hover:text-primary-600"
                            }`}
                          >
                            {item.label}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      onClick={() => setExpandedViz("map")}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                    >
                      Expand
                    </button>
                  </div>
                </div>
                {mapListings.length === 0 ? (
                  <div className="h-[320px] rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 flex items-center justify-center text-sm text-slate-400">
                    No listings match the current filter combination.
                  </div>
                ) : (
                  renderMapVisualization(440)
                )}
                {renderMapLegend()}
              </section>

              {/* Summary cards */}
              <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <h3 className="text-sm font-semibold text-slate-700">Listings in view</h3>
                  <p className="text-xs text-slate-400 mt-1">Listings currently represented after applying filters.</p>
                  <div className="mt-4 grid grid-cols-1 gap-2 text-sm">
                    <div className="flex items-center justify-between rounded-2xl bg-primary-50/70 px-4 py-2">
                      <span className="font-semibold text-primary-700">Highbury</span>
                      <span className="font-mono text-slate-700">
                        {formatNumber(totals?.highburyListings ?? 0, { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-2xl bg-sky-50 px-4 py-2">
                      <span className="font-semibold text-sky-700">{comparisonLabel}</span>
                      <span className="font-mono text-slate-700">
                        {formatNumber(totals?.comparisonListings ?? 0, { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <h3 className="text-sm font-semibold text-slate-700">Average overall rating</h3>
                  <p className="text-xs text-slate-400 mt-1">Aggregated guest rating (0–5 scale).</p>
                  <div className="mt-6 flex flex-col gap-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-primary-700">Highbury</span>
                      <span className="font-mono text-slate-700">
                        {formatNumber(ratingSummary?.highburyAverage ?? null, { maximumFractionDigits: 2 })}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sky-700">{comparisonLabel}</span>
                      <span className="font-mono text-slate-700">
                        {formatNumber(ratingSummary?.comparisonAverage ?? null, { maximumFractionDigits: 2 })}
                      </span>
                    </div>
                  </div>
                </div>
              </section>

              {/* Occupancy & Revenue */}
              <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-700">Occupancy snapshot</h3>
                    <span className="text-xs text-slate-400">Average occupancy outlook.</span>
                  </div>
                  {occupancyCards.length === 0 ? (
                    <p className="text-xs text-slate-400">No listings matched your filters.</p>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {occupancyCards.map((item) => (
                        <div key={item.metric} className="rounded-2xl border border-slate-100 bg-slate-50/70 px-4 py-3">
                          <div className="text-xs uppercase text-slate-400">{item.label}</div>
                          <div className="flex items-center justify-between mt-2 text-sm">
                            <span className="font-semibold text-primary-600">Highbury</span>
                            <span className="font-mono text-slate-700">{formatPercentage(item.highbury, 1)}</span>
                          </div>
                          <div className="flex items-center justify-between text-sm mt-1">
                            <span className="font-semibold text-sky-600">{comparisonLabel}</span>
                            <span className="font-mono text-slate-700">{formatPercentage(item.comparison, 1)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-700">Revenue outlook</h3>
                    <span className="text-xs text-slate-400">Estimated gross revenue for the same periods.</span>
                  </div>
                  {revenueCards.length === 0 ? (
                    <p className="text-xs text-slate-400">No revenue projections for the current filters.</p>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {revenueCards.map((item) => (
                        <div key={item.metric} className="rounded-2xl border border-slate-100 bg-white px-4 py-3 shadow-sm">
                          <div className="text-xs uppercase text-slate-400">{item.label}</div>
                          <div className="flex items-center justify-between mt-2 text-sm">
                            <span className="font-semibold text-primary-600">Highbury</span>
                            <span className="font-mono text-slate-700">
                              {formatNumber(item.highbury, { style: "currency", currency: "USD", maximumFractionDigits: 0 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-sm mt-1">
                            <span className="font-semibold text-sky-600">{comparisonLabel}</span>
                            <span className="font-mono text-slate-700">
                              {formatNumber(item.comparison, { style: "currency", currency: "USD", maximumFractionDigits: 0 })}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>

              {/* Guest experience & distribution */}
              <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-700">Guest experience metrics</h3>
                      <span className="text-xs text-slate-400">Average review subscores (0–5).</span>
                    </div>
                    {/* NEW: Expand button */}
                    <button
                      onClick={() => setExpandedViz("guestExperience")}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                    >
                      Expand
                    </button>
                  </div>
                  {reviewScoreChartData.length === 0 ? <p className="text-xs text-slate-400">No data.</p> : <GuestExperienceChart height={288} />}
                </div>

                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-700">Rating distribution</h3>
                      <div className="flex items-center gap-4 text-xs text-slate-500">
                        <span className="font-semibold text-primary-600">
                          Highbury: {formatNumber(ratingSummary?.highburyAverage ?? null, { maximumFractionDigits: 2 })}
                        </span>
                        <span className="font-semibold text-sky-600">
                          {comparisonLabel}: {formatNumber(ratingSummary?.comparisonAverage ?? null, { maximumFractionDigits: 2 })}
                        </span>
                      </div>
                    </div>
                    {/* NEW: Expand button */}
                    <button
                      onClick={() => setExpandedViz("ratingDistribution")}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                    >
                      Expand
                    </button>
                  </div>
                  {ratingDistributionData.length === 0 ? <p className="text-xs text-slate-400">No data.</p> : <RatingDistributionChart height={288} />}
                </div>
              </section>

              {/* Price + host counts */}
              <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-700">Price summary</h3>
                    <span className="text-xs text-slate-400">Five-number summary (USD).</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-slate-600">
                    <div className="rounded-2xl border border-slate-100 bg-slate-50/80 px-4 py-3">
                      <div className="text-xs uppercase text-slate-400 mb-1">Highbury</div>
                      {PRICE_KEYS.map((key) => (
                        <div key={key} className="flex items-center justify-between text-xs py-0.5">
                          <span className="capitalize text-slate-500">{key}</span>
                          <span className="font-mono">
                            {formatNumber(priceSummary?.highbury?.[key] ?? null, { style: "currency", currency: "USD", maximumFractionDigits: 0 })}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="rounded-2xl border border-slate-100 bg-slate-50/80 px-4 py-3">
                      <div className="text-xs uppercase text-slate-400 mb-1">{comparisonLabel}</div>
                      {PRICE_KEYS.map((key) => (
                        <div key={key} className="flex items-center justify-between text-xs py-0.5">
                          <span className="capitalize text-slate-500">{key}</span>
                          <span className="font-mono">
                            {formatNumber(priceSummary?.comparison?.[key] ?? null, { style: "currency", currency: "USD", maximumFractionDigits: 0 })}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-4 sm:p-6">
                  <h3 className="text-sm font-semibold text-slate-700 mb-3">Host property counts</h3>
                  {combinedHostCounts.length === 0 ? (
                    <p className="text-xs text-slate-400">No hosts in view for the selected filters.</p>
                  ) : (
                    <div className="max-h-72 overflow-y-auto pr-1">
                      <table className="w-full text-xs text-left text-slate-600">
                        <thead className="text-[11px] uppercase text-slate-400">
                          <tr>
                            <th className="py-2">Host</th>
                            <th className="py-2 text-right">Listings</th>
                          </tr>
                        </thead>
                        <tbody>
                          {combinedHostCounts.map((host) => {
                            const isHighbury = host.hostName.toLowerCase() === "highbury";
                            return (
                              <tr key={host.hostName} className="border-t border-slate-100">
                                <td className={`py-2 ${isHighbury ? "font-semibold text-primary-700" : ""}`}>{host.hostName}</td>
                                <td className={`py-2 text-right font-semibold ${isHighbury ? "text-primary-700" : "text-slate-700"}`}>
                                  {host.listings}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </section>
            </>
          ) : (
            <DashboardSkeleton />
          )}
        </main>

        {/* EXPANDED MODALS */}
        {expandedViz && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/70 px-4 py-10" onClick={() => setExpandedViz(null)}>
            <div className="relative w-full max-w-5xl rounded-3xl bg-white shadow-2xl p-6 space-y-4" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between gap-4">
                <div>
                  {expandedViz === "map" && (
                    <>
                      <h2 className="text-lg font-semibold text-slate-800">New York City coverage — full view</h2>
                      <p className="text-xs text-slate-500">
                        Expanded map uses the same sampling logic for market listings (currently showing{" "}
                        {formatNumber(insights?.map.rendered ?? 0, { maximumFractionDigits: 0 })} of{" "}
                        {formatNumber(insights?.map.total ?? 0, { maximumFractionDigits: 0 })} records).
                      </p>
                    </>
                  )}
                  {expandedViz === "guestExperience" && <h2 className="text-lg font-semibold text-slate-800">Guest experience metrics</h2>}
                  {expandedViz === "ratingDistribution" && <h2 className="text-lg font-semibold text-slate-800">Rating distribution</h2>}
                </div>
                <button
                  onClick={() => setExpandedViz(null)}
                  className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                >
                  Close
                </button>
              </div>

              {expandedViz === "map" && (
                <>
                  {mapListings.length === 0 ? <p className="text-sm text-slate-500">No listings match the current filters.</p> : renderMapVisualization(560)}
                  {renderMapLegend()}
                </>
              )}

              {expandedViz === "guestExperience" && (
                <>
                  {reviewScoreChartData.length === 0 ? <p className="text-sm text-slate-500">No data.</p> : <GuestExperienceChart height={420} />}
                </>
              )}

              {expandedViz === "ratingDistribution" && (
                <>
                  {ratingDistributionData.length === 0 ? <p className="text-sm text-slate-500">No data.</p> : <RatingDistributionChart height={420} />}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
