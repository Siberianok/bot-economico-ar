import { apiGet } from "./client";
import type { SessionData } from "../types/api";

export function getSession() {
  return apiGet<SessionData>("/api/v1/session");
}
