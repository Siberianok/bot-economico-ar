import { apiGet } from "./client";
import type { SignalItem } from "../types/api";

export interface SignalsData {
  items: SignalItem[];
}

export function getSignals() {
  return apiGet<SignalsData>("/api/v1/signals");
}

