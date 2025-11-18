declare module "geojson" {
  export type Geometry = any;
  export type GeometryCollection = any;
  export type GeometryObject = any;
  export type Position = number[];
  export type Point = any;
  export type MultiPoint = any;
  export type LineString = any;
  export type MultiLineString = any;
  export type Polygon = any;
  export type MultiPolygon = any;

  export interface Feature<G = any, P = any> {
    type: string;
    geometry: G;
    properties: P;
    bbox?: number[];
    id?: string | number;
  }

  export interface FeatureCollection<G = any, P = any> {
    type: string;
    features: Array<Feature<G, P>>;
    bbox?: number[];
  }

  export interface GeoJsonObject {
    type: string;
    bbox?: number[];
  }
}
