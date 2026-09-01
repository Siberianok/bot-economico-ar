import { BarChart, PieChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/esm/core";

echarts.use([
  BarChart,
  PieChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export interface PortfolioChartItem {
  label: string;
  value: number;
}

const palette = ["#38BDF8", "#8B5CF6", "#22C55E", "#FACC15", "#EF4444", "#8FA6C2"];

export function PortfolioDonutChart({ items }: { items: PortfolioChartItem[] }) {
  const option = {
    backgroundColor: "transparent",
    color: palette,
    tooltip: { trigger: "item" },
    legend: {
      type: "scroll",
      bottom: 0,
      textStyle: { color: "#8FA6C2" },
    },
    series: [
      {
        type: "pie",
        radius: ["48%", "72%"],
        center: ["50%", "43%"],
        avoidLabelOverlap: true,
        label: { show: false },
        emphasis: { label: { show: true, color: "#EAF2FF", fontWeight: 600 } },
        data: items.map((item) => ({ name: item.label, value: item.value })),
      },
    ],
  };
  return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 280, width: "100%" }} />;
}

export function PortfolioResultChart({ items }: { items: PortfolioChartItem[] }) {
  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    grid: { left: 20, right: 20, top: 12, bottom: 28, containLabel: true },
    xAxis: {
      type: "value",
      axisLabel: { color: "#8FA6C2" },
      splitLine: { lineStyle: { color: "#1F3554" } },
    },
    yAxis: {
      type: "category",
      data: items.map((item) => item.label),
      axisLabel: { color: "#8FA6C2" },
      axisLine: { lineStyle: { color: "#1F3554" } },
    },
    series: [
      {
        type: "bar",
        data: items.map((item) => ({
          value: item.value,
          itemStyle: {
            color: item.value >= 0 ? "#22C55E" : "#EF4444",
            borderRadius: item.value >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4],
          },
        })),
        barMaxWidth: 24,
      },
    ],
  };
  return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 280, width: "100%" }} />;
}
