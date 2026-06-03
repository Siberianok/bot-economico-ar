import { apiGet } from "./client";
import type { Block } from "../types/api";

export interface MarketPulseData {
  dolar_oficial: Block;
  dolar_blue: Block;
  dolar_mep: Block;
  dolar_ccl: Block;
  dolar_cripto: Block;
  reservas: Block;
  inflacion: Block;
  riesgo_pais: Block;
  bandas_cambiarias: Block;
  brechas: Block;
  signals: Block;
  news: Block;
  calendar: Block;
}

export function getMarketPulse() {
  return apiGet<MarketPulseData>("/api/v1/market/pulse");
}

