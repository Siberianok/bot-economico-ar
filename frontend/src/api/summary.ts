import { apiGet } from "./client";
import type { SummaryData } from "../types/api";

export function getDailySummary() {
  return apiGet<SummaryData>("/api/v1/summary");
}
