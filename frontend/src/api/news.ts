import { apiGet } from "./client";
import type { NewsData } from "../types/api";

export function getNews() {
  return apiGet<NewsData>("/api/v1/news?limit=18");
}
