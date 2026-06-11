import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  api,
  type CatalogueCoffeeSummary,
  type LandscapeResponse,
  type RecommendationPayload,
  type ReviewedCoffeeDetails,
  type ReviewSessionPayload,
} from "./lib/api";

type InputMode = "catalogue" | "url";

const LandscapePlot = lazy(() => import("./LandscapePlot"));

const defaultReview =
  "Silky and sweet, but I want a touch less acidity and a little more chocolate depth.";

const processLabels: Record<string, string> = {
  process_washed: "Washed",
  process_natural: "Natural",
  process_honey: "Honey",
  process_anaerobic: "Anaerobic",
  process_cofermented: "Co-fermented",
};

function sentenceCase(key: string): string {
  return key
    .replace(/^process_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function getTopSensoryNotes(sensory: Record<string, number>): string[] {
  return Object.entries(sensory)
    .sort(([, left], [, right]) => right - left)
    .slice(0, 4)
    .map(([key, value]) => `${sentenceCase(key)} ${formatPercent(value)}`);
}

function getPrimaryProcess(process: Record<string, number>): string {
  const [primary] = Object.entries(process).sort(([, left], [, right]) => right - left);
  if (!primary || primary[1] <= 0) {
    return "Unknown";
  }
  return processLabels[primary[0]] || sentenceCase(primary[0]);
}

function formatProcessLabel(process: string | null): string | null {
  if (!process) {
    return null;
  }
  return sentenceCase(process);
}

function buildPlotlyTheme(figure: Record<string, unknown>): Record<string, unknown> {
  const layout = (figure.layout as Record<string, unknown> | undefined) || {};
  const scene = (layout.scene as Record<string, unknown> | undefined) || {};
  const nextLayout: Record<string, unknown> = {
    ...layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(246, 236, 221, 0.58)",
    font: {
      family: '"Instrument Sans", sans-serif',
      color: "#4a3427",
    },
    margin: {
      l: 40,
      r: 20,
      t: 32,
      b: 40,
      ...(layout.margin || {}),
    },
  };

  if (layout.xaxis) {
    nextLayout.xaxis = {
      ...layout.xaxis,
      gridcolor: "rgba(121, 92, 69, 0.12)",
      zerolinecolor: "rgba(121, 92, 69, 0.18)",
    };
  }

  if (layout.yaxis) {
    nextLayout.yaxis = {
      ...layout.yaxis,
      gridcolor: "rgba(121, 92, 69, 0.12)",
      zerolinecolor: "rgba(121, 92, 69, 0.18)",
    };
  }

  if (layout.scene) {
    nextLayout.scene = {
      ...scene,
      bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        ...((scene.xaxis as Record<string, unknown> | undefined) || {}),
        backgroundcolor: "rgba(246, 236, 221, 0.2)",
        gridcolor: "rgba(121, 92, 69, 0.14)",
        zerolinecolor: "rgba(121, 92, 69, 0.2)",
      },
      yaxis: {
        ...((scene.yaxis as Record<string, unknown> | undefined) || {}),
        backgroundcolor: "rgba(246, 236, 221, 0.2)",
        gridcolor: "rgba(121, 92, 69, 0.14)",
        zerolinecolor: "rgba(121, 92, 69, 0.2)",
      },
      zaxis: {
        ...((scene.zaxis as Record<string, unknown> | undefined) || {}),
        backgroundcolor: "rgba(246, 236, 221, 0.2)",
        gridcolor: "rgba(121, 92, 69, 0.14)",
        zerolinecolor: "rgba(121, 92, 69, 0.2)",
      },
    };
  }

  return {
    ...figure,
    layout: nextLayout,
  };
}

function App() {
  const [catalogueCoffees, setCatalogueCoffees] = useState<CatalogueCoffeeSummary[]>([]);
  const [reviewSession, setReviewSession] = useState<ReviewSessionPayload | null>(null);
  const [landscape, setLandscape] = useState<LandscapeResponse | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>("catalogue");
  const [selectedCoffeeId, setSelectedCoffeeId] = useState("");
  const [selectedReviewedCoffee, setSelectedReviewedCoffee] = useState<ReviewedCoffeeDetails | null>(null);
  const [urlValue, setUrlValue] = useState("");
  const [reviewText, setReviewText] = useState(defaultReview);
  const [topK, setTopK] = useState(5);
  const [showSurface, setShowSurface] = useState(true);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [processingUrl, setProcessingUrl] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadInitialState() {
      setLoading(true);
      setErrorMessage(null);

      try {
        const [coffees, session] = await Promise.all([
          api.listCatalogueCoffees(),
          api.getReviewSession(),
        ]);

        if (!active) {
          return;
        }

        setCatalogueCoffees(coffees);
        setReviewSession(session);

        if (coffees.length > 0) {
          const firstCoffeeId = coffees[0].coffee_id;
          setSelectedCoffeeId(firstCoffeeId);
          const reviewed = await api.getCatalogueReviewedCoffee(firstCoffeeId);
          if (!active) {
            return;
          }
          setSelectedReviewedCoffee(reviewed);
        }
      } catch (error) {
        if (!active) {
          return;
        }
        setErrorMessage(error instanceof Error ? error.message : "Could not load the app.");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadInitialState();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!reviewSession?.review_events.length) {
      setLandscape(null);
      return;
    }

    let active = true;

    async function loadLandscape() {
      try {
        const nextLandscape = await api.buildLandscape(showSurface);
        if (active) {
          setLandscape(nextLandscape);
        }
      } catch (error) {
        if (active) {
          setErrorMessage(error instanceof Error ? error.message : "Could not load the landscape.");
        }
      }
    }

    void loadLandscape();

    return () => {
      active = false;
    };
  }, [reviewSession, showSurface]);

  async function handleCatalogueSelection(coffeeId: string) {
    setSelectedCoffeeId(coffeeId);
    setErrorMessage(null);

    try {
      const reviewed = await api.getCatalogueReviewedCoffee(coffeeId);
      setSelectedReviewedCoffee(reviewed);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not load that coffee.");
    }
  }

  async function handleProcessUrl() {
    if (!urlValue.trim()) {
      setErrorMessage("Paste a coffee product URL before processing it.");
      return;
    }

    setProcessingUrl(true);
    setErrorMessage(null);

    try {
      const reviewed = await api.getReviewedCoffeeFromUrl(urlValue.trim());
      setSelectedReviewedCoffee(reviewed);
      setUrlValue(reviewed.normalized_url || urlValue.trim());
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not process that URL.");
    } finally {
      setProcessingUrl(false);
    }
  }

  async function handleSubmitReview() {
    if (!selectedReviewedCoffee) {
      setErrorMessage("Choose a coffee before asking for recommendations.");
      return;
    }

    setSubmitting(true);
    setErrorMessage(null);

    try {
      const response = await api.submitReview({
        review_text: reviewText,
        reviewed_coffee: selectedReviewedCoffee,
        top_k: topK,
      });
      setReviewSession(response.review_session);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not generate recommendations.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleClearSession() {
    setClearing(true);
    setErrorMessage(null);

    try {
      const session = await api.clearReviewSession();
      setReviewSession(session);
      setLandscape(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not clear the tasting session.");
    } finally {
      setClearing(false);
    }
  }

  const topSensoryNotes = useMemo(() => {
    if (!selectedReviewedCoffee) {
      return [];
    }
    return getTopSensoryNotes(selectedReviewedCoffee.features.sensory);
  }, [selectedReviewedCoffee]);

  const recommendations: RecommendationPayload[] = reviewSession?.last_recommendations || [];
  const plotFigure = landscape?.figure ? buildPlotlyTheme(landscape.figure) : null;

  const plotData = (plotFigure?.data || []) as unknown[];
  const plotLayout = (plotFigure?.layout || {}) as Record<string, unknown>;
  const plotConfig = {
    displayModeBar: false,
    responsive: true,
  };

  if (loading) {
    return (
      <main className="shell">
        <section className="loading-card">
          <p className="eyebrow">Coffee Recommender</p>
          <h1>Warming up the tasting room...</h1>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <div className="backdrop backdrop-one" />
      <div className="backdrop backdrop-two" />

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Coffee Recommender</p>
          <h1>Find your next cup with a softer, more editorial tasting experience.</h1>
          <p className="lede">
            Start from a coffee you know, describe what you loved or wanted to change, and let the
            recommender guide you toward something deliciously closer.
          </p>
        </div>

        <div className="hero-aside">
          <div className="stat-card">
            <span className="stat-label">Session picks</span>
            <strong>{recommendations.length || 0}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Tasting notes</span>
            <strong>{reviewSession?.review_events.length || 0}</strong>
          </div>
        </div>
      </section>

      {errorMessage ? (
        <section className="message-banner" role="alert">
          {errorMessage}
        </section>
      ) : null}

      <section className="grid">
        <article className="panel panel-form">
          <div className="panel-heading">
            <p className="eyebrow">1. Choose a starting cup</p>
            <h2>Set the tasting reference</h2>
          </div>

          <div className="segmented-control" role="tablist" aria-label="Reviewed coffee source">
            <button
              className={inputMode === "catalogue" ? "active" : ""}
              onClick={() => setInputMode("catalogue")}
              type="button"
            >
              Catalogue coffee
            </button>
            <button
              className={inputMode === "url" ? "active" : ""}
              onClick={() => setInputMode("url")}
              type="button"
            >
              Coffee URL
            </button>
          </div>

          {inputMode === "catalogue" ? (
            <label className="field">
              <span>Reviewed coffee</span>
              <select
                value={selectedCoffeeId}
                onChange={(event) => {
                  void handleCatalogueSelection(event.target.value);
                }}
              >
                {catalogueCoffees.map((coffee) => (
                  <option key={coffee.coffee_id} value={coffee.coffee_id}>
                    {coffee.name || coffee.coffee_id}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="field-group">
              <label className="field">
                <span>Coffee product URL</span>
                <input
                  type="url"
                  value={urlValue}
                  placeholder="https://..."
                  onChange={(event) => setUrlValue(event.target.value)}
                />
              </label>
              <button className="secondary-button" onClick={() => void handleProcessUrl()} type="button">
                {processingUrl ? "Processing..." : "Process coffee"}
              </button>
            </div>
          )}

          <div className="panel-heading panel-heading-tight">
            <p className="eyebrow">2. Describe the cup</p>
            <h2>Write your tasting note</h2>
          </div>

          <label className="field">
            <span>Your review</span>
            <textarea
              value={reviewText}
              onChange={(event) => setReviewText(event.target.value)}
              rows={7}
            />
          </label>

          <div className="settings-row">
            <label className="field">
              <span>How many recommendations</span>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
              <small>{topK} coffees</small>
            </label>

            <div className="actions">
              <button className="primary-button" onClick={() => void handleSubmitReview()} type="button">
                {submitting ? "Brewing suggestions..." : "Add review and recommend"}
              </button>
              <button className="ghost-button" onClick={() => void handleClearSession()} type="button">
                {clearing ? "Clearing..." : "Clear tasting session"}
              </button>
            </div>
          </div>
        </article>

        <article className="panel panel-spotlight">
          <div className="panel-heading">
            <p className="eyebrow">Current cup</p>
            <h2>{selectedReviewedCoffee?.features.name || "Choose a coffee to begin"}</h2>
          </div>

          {selectedReviewedCoffee ? (
            <>
              <div className="pill-row">
                <span className="pill">{getPrimaryProcess(selectedReviewedCoffee.features.process)}</span>
                <span className="pill">{selectedReviewedCoffee.source_type === "catalogue" ? "Catalogue" : "URL imported"}</span>
              </div>

              <p className="spotlight-copy">
                The recommender will use this cup as the anchor, then tilt future suggestions toward
                the changes you describe in your note.
              </p>

              <div className="note-grid">
                {topSensoryNotes.map((note) => (
                  <div className="note-card" key={note}>
                    {note}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="spotlight-copy">
              Pick a coffee from the catalogue or bring in a product URL to start shaping the
              recommendations.
            </p>
          )}
        </article>
      </section>

      <section className="results-section">
        <div className="section-heading">
          <p className="eyebrow">Recommendations</p>
          <h2>What to brew next</h2>
        </div>

        {recommendations.length === 0 ? (
          <article className="panel empty-state">
            Add a tasting note to reveal your first ranked set of coffees.
          </article>
        ) : (
          <div className="recommendation-grid">
            {recommendations.map((recommendation, index) => (
              <article className="panel recommendation-card" key={recommendation.coffee_id}>
                <span className="recommendation-rank">#{index + 1}</span>
                <h3>{recommendation.name || recommendation.coffee_id}</h3>
                <p className="recommendation-meta">
                  {recommendation.roaster || recommendation.producer || recommendation.coffee_id}
                </p>
                <div className="recommendation-detail-row">
                  {recommendation.origin_country ? (
                    <span className="mini-pill">{recommendation.origin_country}</span>
                  ) : null}
                  {recommendation.process ? (
                    <span className="mini-pill">{formatProcessLabel(recommendation.process)}</span>
                  ) : null}
                </div>
                {recommendation.tasting_notes.length ? (
                  <p className="recommendation-notes">
                    {recommendation.tasting_notes.slice(0, 3).join(" • ")}
                  </p>
                ) : null}
                <div className="score-row">
                  <div>
                    <span>Match score</span>
                    <strong>{recommendation.score.toFixed(3)}</strong>
                  </div>
                </div>
                {recommendation.source_url ? (
                  <a
                    className="recommendation-link"
                    href={recommendation.source_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View coffee
                  </a>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="results-section">
        <div className="section-heading landscape-header">
          <div>
            <p className="eyebrow">Landscape</p>
            <h2>Where your taste sits in the coffee map</h2>
          </div>

          <label className="toggle">
            <input
              type="checkbox"
              checked={showSurface}
              onChange={(event) => setShowSurface(event.target.checked)}
            />
            <span>Show score surface</span>
          </label>
        </div>

        {!reviewSession?.review_events.length ? (
          <article className="panel empty-state">
            Add at least one review to place your preferences on the projected coffee landscape.
          </article>
        ) : landscape?.message ? (
          <article className="panel empty-state">{landscape.message}</article>
        ) : plotFigure ? (
          <article className="panel chart-panel">
            <Suspense fallback={<div className="plot-loading">Loading the map...</div>}>
              <LandscapePlot
                className="plot"
                data={plotData}
                layout={plotLayout}
                config={plotConfig}
              />
            </Suspense>
          </article>
        ) : (
          <article className="panel empty-state">Loading the landscape...</article>
        )}
      </section>
    </main>
  );
}

export default App;
