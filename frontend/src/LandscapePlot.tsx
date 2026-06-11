import Plot from "react-plotly.js";

type LandscapePlotProps = {
  className?: string;
  data: unknown[];
  layout: Record<string, unknown>;
  config: Record<string, unknown>;
};

export default function LandscapePlot({
  className,
  data,
  layout,
  config,
}: LandscapePlotProps) {
  return (
    <Plot
      className={className}
      data={data}
      layout={layout}
      config={config}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
