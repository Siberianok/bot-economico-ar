import { apiGet } from "./client";
import type { PortfolioSummary } from "../types/api";

export function getPortfolioSummary() {
  return apiGet<PortfolioSummary>("/api/v1/portfolio/summary");
}
