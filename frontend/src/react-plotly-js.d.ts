declare module "react-plotly.js" {
  import type { CSSProperties } from "react";

  type PlotProps = {
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    className?: string;
    style?: CSSProperties;
    useResizeHandler?: boolean;
  };

  export default function Plot(props: PlotProps): JSX.Element;
}
