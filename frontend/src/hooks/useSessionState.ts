import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

export function useSessionState<T>(key: string, initialValue: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const saved = window.sessionStorage.getItem(key);
      return saved ? (JSON.parse(saved) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore cache write failures. The page can still work without session cache.
    }
  }, [key, value]);

  return [value, setValue];
}
