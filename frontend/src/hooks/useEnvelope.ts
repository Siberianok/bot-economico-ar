import { useEffect, useState } from "react";
import type { ApiEnvelope } from "../types/api";

export function useEnvelope<T>(loader: () => Promise<ApiEnvelope<T>>, deps: unknown[] = []) {
  const [payload, setPayload] = useState<ApiEnvelope<T> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    loader()
      .then((result) => {
        if (mounted) setPayload(result);
      })
      .catch((err) => {
        if (mounted) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, deps);

  return { payload, loading, error };
}

