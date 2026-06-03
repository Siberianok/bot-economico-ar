import { useState } from "react";
import { AppShell, type PageKey } from "./components/Layout/AppShell";
import { Alerts } from "./pages/Alerts";
import { MarketPulse } from "./pages/MarketPulse";
import { PortfolioCommandCenter } from "./pages/PortfolioCommandCenter";
import { Screener } from "./pages/Screener";
import { Settings } from "./pages/Settings";
import { Signals } from "./pages/Signals";

export default function App() {
  const [active, setActive] = useState<PageKey>("market");
  const pages: Record<PageKey, JSX.Element> = {
    market: <MarketPulse />,
    screener: <Screener />,
    portfolio: <PortfolioCommandCenter />,
    alerts: <Alerts />,
    signals: <Signals />,
    settings: <Settings />,
  };
  return (
    <AppShell active={active} onNavigate={setActive}>
      {pages[active]}
    </AppShell>
  );
}

