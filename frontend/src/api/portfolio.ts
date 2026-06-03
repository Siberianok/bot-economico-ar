import { apiGet } from "./client";
import type { Block } from "../types/api";

export type PortfolioSummary = Record<string, Block>;

export function getPortfolioSummary() {
  return apiGet<PortfolioSummary>("/api/v1/portfolio/summary");
}

