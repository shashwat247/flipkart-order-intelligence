import { useEffect, useState } from "react";

/** Hand-rolled hash router — no router dependency, per the spec's "no other
 * runtime dependencies" rule beyond recharts/lucide-react. */
export function useHashRoute(): [string, (path: string) => void] {
  const [route, setRoute] = useState(() => normalize(window.location.hash));

  useEffect(() => {
    const onChange = () => setRoute(normalize(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = (path: string) => {
    window.location.hash = path;
  };

  return [route, navigate];
}

function normalize(hash: string): string {
  const path = hash.replace(/^#/, "");
  return path || "/assistant";
}
