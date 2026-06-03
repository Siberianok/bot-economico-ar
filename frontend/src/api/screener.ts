import { apiGet } from "./client";
import type { ScreenerItem } from "../types/api";

export interface ScreenerData {
  kind: "acciones" | "cedears";
  items: ScreenerItem[];
}

export function getScreener(kind: "acciones" | "cedears") {
  return apiGet<ScreenerData>(`/api/v1/screener?kind=${kind}`);
}

