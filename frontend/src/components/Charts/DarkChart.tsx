import ReactECharts from "echarts-for-react";

interface Props {
  title: string;
  labels: string[];
  values: number[];
}

export function DarkChart({ title, labels, values }: Props) {
  const option = {
    backgroundColor: "transparent",
    title: { text: title, textStyle: { color: "#EAF2FF", fontSize: 13 } },
    tooltip: { trigger: "axis" },
    grid: { left: 36, right: 18, top: 48, bottom: 32 },
    xAxis: {
      type: "category",
      data: labels,
      axisLine: { lineStyle: { color: "#1F3554" } },
      axisLabel: { color: "#8FA6C2" },
    },
    yAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#1F3554" } },
      splitLine: { lineStyle: { color: "#1F3554" } },
      axisLabel: { color: "#8FA6C2" },
    },
    series: [
      {
        type: "bar",
        data: values,
        itemStyle: { color: "#38BDF8", borderRadius: [4, 4, 0, 0] },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 280, width: "100%" }} />;
}

