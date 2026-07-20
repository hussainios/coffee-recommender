import { startTransition, useEffect, useState } from "react";
import {
  api,
  type CatalogueCoffeeSummary,
  type CoffeeDetailPayload,
  type RecommendationPayload,
  type ReviewedCoffeeDetails,
  type ReviewSessionPayload,
} from "./lib/api";

type Screen = "home" | "detail" | "review";
type ReviewMood = "loved" | "liked" | "okay" | "nope";

type FeedbackChip = {
  id: string;
  label: string;
  phrase: string;
};

const feedbackChips: FeedbackChip[] = [
  { id: "floral", label: "More floral", phrase: "I want something a little more floral." },
  { id: "sweet", label: "Sweeter", phrase: "I want a sweeter cup." },
  { id: "body", label: "More body", phrase: "I want a touch more body." },
  { id: "less_acid", label: "Less acidic", phrase: "I want slightly lower acidity." },
  { id: "cleaner", label: "Cleaner", phrase: "I want something cleaner and more transparent." },
  { id: "less_funky", label: "Less funky", phrase: "I want less fermentation and funk." },
];

const moodCopy: Record<ReviewMood, { title: string; value: number; sentence: string }> = {
  loved: {
    title: "Loved it",
    value: 1,
    sentence: "I loved this coffee.",
  },
  liked: {
    title: "Liked it",
    value: 0.45,
    sentence: "I liked this coffee overall.",
  },
  okay: {
    title: "It was okay",
    value: 0,
    sentence: "This coffee was okay, but not a standout.",
  },
  nope: {
    title: "Not for me",
    value: -0.7,
    sentence: "This coffee was not really for me.",
  },
};

function formatPercent(score: number): string {
  return `${Math.round(score * 100)}% match`;
}

function formatProcess(process: string | null): string | null {
  if (!process) {
    return null;
  }

  return process
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function buildReasonLine(coffee: RecommendationPayload | CatalogueCoffeeSummary): string {
  const notes = coffee.tasting_notes?.slice(0, 2) || [];
  const descriptors = [coffee.origin_country, formatProcess(coffee.process ?? null)]
    .filter(Boolean)
    .join(" · ");

  if (notes.length) {
    return `${descriptors ? `${descriptors} · ` : ""}${notes.join(" · ")}`;
  }

  return descriptors || "A strong fit for your current taste profile";
}

function buildWhyRecommended(recommendation: RecommendationPayload | null, detail: CoffeeDetailPayload | null): string {
  if (recommendation?.tasting_notes.length) {
    return `Because you’ve been leaning toward coffees with ${recommendation.tasting_notes
      .slice(0, 3)
      .join(", ")}.`;
  }

  if (detail?.tasting_notes.length) {
    return `Because this cup should land around ${detail.tasting_notes.slice(0, 3).join(", ")}.`;
  }

  if (detail?.origin_country || detail?.process) {
    return `Because it sits in the same zone as coffees you’ve been responding well to recently.`;
  }

  return "Because it should be a natural next step from the coffees you’ve liked so far.";
}

function getFlavorBars(detail: CoffeeDetailPayload | null): Array<{ label: string; value: number }> {
  if (!detail) {
    return [];
  }

  const sensory = detail.features.sensory;
  return [
    { label: "Acidity", value: sensory.acidity ?? 0.5 },
    { label: "Sweetness", value: sensory.sweetness ?? 0.5 },
    { label: "Body", value: sensory.body ?? 0.5 },
    { label: "Floral", value: sensory.floral ?? sensory.fruitiness ?? 0.5 },
  ];
}

function buildReviewText(mood: ReviewMood, selectedChipIds: string[], note: string): string {
  const parts = [moodCopy[mood].sentence];
  const chipPhrases = feedbackChips
    .filter((chip) => selectedChipIds.includes(chip.id))
    .map((chip) => chip.phrase);

  if (chipPhrases.length) {
    parts.push(chipPhrases.join(" "));
  }

  if (note.trim()) {
    parts.push(note.trim());
  }

  return parts.join(" ");
}

function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [catalogueCoffees, setCatalogueCoffees] = useState<CatalogueCoffeeSummary[]>([]);
  const [reviewSession, setReviewSession] = useState<ReviewSessionPayload | null>(null);
  const [selectedCoffeeId, setSelectedCoffeeId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<CoffeeDetailPayload | null>(null);
  const [selectedReviewedCoffee, setSelectedReviewedCoffee] = useState<ReviewedCoffeeDetails | null>(null);
  const [reviewMood, setReviewMood] = useState<ReviewMood>("liked");
  const [selectedChipIds, setSelectedChipIds] = useState<string[]>([]);
  const [reviewNote, setReviewNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
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

    void load();

    return () => {
      active = false;
    };
  }, []);

  const recommendations = reviewSession?.last_recommendations || [];
  const primaryCards =
    recommendations.length > 0 ? recommendations : catalogueCoffees.slice(0, 6);
  const selectedRecommendation =
    recommendations.find((coffee) => coffee.coffee_id === selectedCoffeeId) || null;
  const latestReviewCount = reviewSession?.review_events.length || 0;

  async function openCoffee(coffeeId: string) {
    setDetailLoading(true);
    setErrorMessage(null);

    try {
      const [detail, reviewedCoffee] = await Promise.all([
        api.getCoffeeDetail(coffeeId),
        api.getCatalogueReviewedCoffee(coffeeId),
      ]);

      startTransition(() => {
        setSelectedCoffeeId(coffeeId);
        setSelectedDetail(detail);
        setSelectedReviewedCoffee(reviewedCoffee);
        setScreen("detail");
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not load that coffee.");
    } finally {
      setDetailLoading(false);
    }
  }

  function moveToReview() {
    setReviewMood("liked");
    setSelectedChipIds([]);
    setReviewNote("");
    setScreen("review");
  }

  function toggleChip(chipId: string) {
    setSelectedChipIds((current) =>
      current.includes(chipId) ? current.filter((value) => value !== chipId) : [...current, chipId],
    );
  }

  async function submitReview() {
    if (!selectedReviewedCoffee || !selectedCoffeeId) {
      setErrorMessage("Choose a coffee before submitting feedback.");
      return;
    }

    setSubmitting(true);
    setErrorMessage(null);
    setStatusMessage(null);

    try {
      const response = await api.submitReview({
        review_text: buildReviewText(reviewMood, selectedChipIds, reviewNote),
        reviewed_coffee: selectedReviewedCoffee,
        top_k: 5,
      });

      setReviewSession(response.review_session);
      setStatusMessage("Recommendations refreshed for your next cup.");
      setScreen("home");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not update recommendations.");
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <main className="app-shell">
        <section className="phone-frame loading-state">
          <p className="eyebrow">Coffee Recommender</p>
          <h1>Brewing your pocket tasting room...</h1>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-top" />
      <div className="ambient ambient-bottom" />

      <section className="phone-frame">
        <header className="topbar">
          {screen === "home" ? (
            <>
              <div className="brand-block">
                <div className="brand-row">
                  <span className="brand-mark" aria-hidden="true" />
                  <p className="eyebrow">Coffee Recommender</p>
                </div>
                <h1>For you</h1>
                <p className="topbar-subtitle">
                  {recommendations.length > 0
                    ? "Fresh picks shaped by your recent feedback"
                    : "Start with a coffee you know and we’ll sharpen the next recs."}
                </p>
              </div>
              <span className="topbar-pill">
                {latestReviewCount > 0 ? `${latestReviewCount} reviews` : "Start tasting"}
              </span>
            </>
          ) : (
            <>
              <button className="icon-button" onClick={() => setScreen("home")} type="button">
                Back
              </button>
              <div className="topbar-copy">
                <p className="eyebrow">{screen === "detail" ? "Coffee detail" : "Review"}</p>
                <h1>{screen === "detail" ? "The next cup" : "How was it?"}</h1>
              </div>
            </>
          )}
        </header>

        {errorMessage ? <div className="banner banner-error">{errorMessage}</div> : null}
        {statusMessage && !errorMessage ? <div className="banner">{statusMessage}</div> : null}

        {screen === "home" ? (
          <section className="screen-content home-screen">
            <div className="home-intro card">
              <div className="home-intro-copy">
                <p className="section-kicker">
                  {recommendations.length > 0 ? "Recommended next" : "First step"}
                </p>
                <h2>
                  {recommendations.length > 0
                    ? "A shortlist that should feel closer to your taste."
                    : "Pick one coffee to teach the app what you like."}
                </h2>
              </div>
              <p className="home-intro-text">
                {recommendations.length > 0
                  ? "Open a coffee, decide if it’s worth trying, then send one fast reaction to refresh the next set."
                  : "You only need one review to start getting better recs."}
              </p>
              <div className="home-intro-stats">
                <div>
                  <span className="stat-label">Available now</span>
                  <strong>{primaryCards.length}</strong>
                </div>
                <div>
                  <span className="stat-label">Feedback logged</span>
                  <strong>{latestReviewCount}</strong>
                </div>
              </div>
            </div>

            <div className="card-stack">
              {primaryCards.map((coffee, index) => {
                const isRecommendation = "score" in coffee;
                const recommendation = isRecommendation ? (coffee as RecommendationPayload) : null;
                return (
                  <article
                    className={`coffee-card ${index === 0 ? "coffee-card-featured" : ""} ${recommendation ? "coffee-card-recommended" : ""}`}
                    key={coffee.coffee_id}
                  >
                    <div className="coffee-card-header">
                      <div>
                        <p className="card-label">{index === 0 ? (isRecommendation ? "Top pick" : "Starter coffee") : "Also worth a look"}</p>
                        <h3>{coffee.name || coffee.coffee_id}</h3>
                        <p className="meta-line">{coffee.roaster || "Roaster unknown"}</p>
                      </div>
                      {recommendation ? (
                        <span className="match-pill">{formatPercent(recommendation.score)}</span>
                      ) : null}
                    </div>

                    <p className="descriptor-line">{buildReasonLine(coffee)}</p>

                    <div className="chip-row">
                      {coffee.tasting_notes.slice(0, 3).map((note) => (
                        <span className="chip" key={note}>
                          {note}
                        </span>
                      ))}
                    </div>

                    <div className="card-actions">
                      <span className="why-line">
                        {recommendation ? "Why you might like it" : "Good starting reference"}
                      </span>
                      <button className="primary-button" onClick={() => void openCoffee(coffee.coffee_id)} type="button">
                        {detailLoading && selectedCoffeeId === coffee.coffee_id ? "Opening..." : "See details"}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        ) : null}

        {screen === "detail" ? (
          <section className="screen-content detail-screen">
            {selectedDetail ? (
              <>
                <article className="detail-hero card">
                  <p className="section-kicker">Coffee detail</p>
                  <h2>{selectedDetail.name || selectedDetail.coffee_id}</h2>
                  <p className="meta-line">{selectedDetail.roaster || "Roaster unknown"}</p>

                  <div className="chip-row">
                    {[selectedDetail.origin_country, formatProcess(selectedDetail.process), selectedDetail.roast_level]
                      .filter(Boolean)
                      .map((item) => (
                        <span className="chip chip-soft" key={item}>
                          {item}
                        </span>
                      ))}
                  </div>
                </article>

                <article className="card detail-section">
                  <p className="section-kicker">At a glance</p>
                  <div className="facts-grid">
                    {selectedDetail.region ? (
                      <div className="fact-item">
                        <span>Region</span>
                        <strong>{selectedDetail.region}</strong>
                      </div>
                    ) : null}
                    {selectedDetail.producer ? (
                      <div className="fact-item">
                        <span>Producer</span>
                        <strong>{selectedDetail.producer}</strong>
                      </div>
                    ) : null}
                    {selectedDetail.price ? (
                      <div className="fact-item">
                        <span>Price</span>
                        <strong>
                          {selectedDetail.currency === "GBP" ? "£" : ""}
                          {selectedDetail.price.toFixed(2)}
                        </strong>
                      </div>
                    ) : null}
                    {selectedDetail.weight_g ? (
                      <div className="fact-item">
                        <span>Bag size</span>
                        <strong>{selectedDetail.weight_g}g</strong>
                      </div>
                    ) : null}
                  </div>
                </article>

                <article className="card detail-section">
                  <p className="section-kicker">Tasting notes</p>
                  <div className="chip-row">
                    {selectedDetail.tasting_notes.length > 0 ? (
                      selectedDetail.tasting_notes.map((note) => (
                        <span className="chip" key={note}>
                          {note}
                        </span>
                      ))
                    ) : (
                      <span className="empty-copy">No tasting notes captured yet.</span>
                    )}
                  </div>
                </article>

                <article className="card detail-section">
                  <p className="section-kicker">Why this one</p>
                  <p className="body-copy">{buildWhyRecommended(selectedRecommendation, selectedDetail)}</p>
                </article>

                <article className="card detail-section">
                  <p className="section-kicker">Flavor profile</p>
                  <div className="flavor-bars">
                    {getFlavorBars(selectedDetail).map((bar) => (
                      <div className="flavor-row" key={bar.label}>
                        <div className="flavor-label-row">
                          <span>{bar.label}</span>
                          <strong>{Math.round(bar.value * 100)}%</strong>
                        </div>
                        <div className="flavor-track">
                          <div className="flavor-fill" style={{ width: `${Math.max(10, Math.round(bar.value * 100))}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </article>

                {selectedDetail.description ? (
                  <article className="card detail-section">
                    <p className="section-kicker">About this coffee</p>
                    <p className="body-copy">{selectedDetail.description}</p>
                  </article>
                ) : null}

                <footer className="sticky-actions">
                  <button className="secondary-button" onClick={() => setScreen("home")} type="button">
                    Keep browsing
                  </button>
                  <button className="primary-button" onClick={moveToReview} type="button">
                    Tried it
                  </button>
                </footer>
              </>
            ) : (
              <article className="card empty-state">Choose a coffee to see its detail.</article>
            )}
          </section>
        ) : null}

        {screen === "review" ? (
          <section className="screen-content review-screen">
            {selectedDetail ? (
              <>
                <article className="card compact-coffee-card">
                  <p className="section-kicker">Reviewing now</p>
                  <h2>{selectedDetail.name || selectedDetail.coffee_id}</h2>
                  <p className="meta-line">
                    {[selectedDetail.roaster, selectedDetail.origin_country, formatProcess(selectedDetail.process)]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                </article>

                <article className="card review-section">
                  <p className="section-kicker">Overall</p>
                  <p className="body-copy">Keep this quick. One reaction is enough to improve the next round.</p>
                  <div className="mood-grid">
                    {Object.entries(moodCopy).map(([key, value]) => (
                      <button
                        className={`mood-button ${reviewMood === key ? "mood-button-active" : ""}`}
                        key={key}
                        onClick={() => setReviewMood(key as ReviewMood)}
                        type="button"
                      >
                        {value.title}
                      </button>
                    ))}
                  </div>
                </article>

                <article className="card review-section">
                  <p className="section-kicker">What would you change?</p>
                  <div className="chip-row">
                    {feedbackChips.map((chip) => (
                      <button
                        className={`filter-chip ${selectedChipIds.includes(chip.id) ? "filter-chip-active" : ""}`}
                        key={chip.id}
                        onClick={() => toggleChip(chip.id)}
                        type="button"
                      >
                        {chip.label}
                      </button>
                    ))}
                  </div>
                </article>

                <article className="card review-section">
                  <label className="field-label" htmlFor="review-note">
                    Optional note
                  </label>
                  <textarea
                    id="review-note"
                    className="note-input"
                    onChange={(event) => setReviewNote(event.target.value)}
                    placeholder="Anything else? One sentence is enough."
                    value={reviewNote}
                  />
                </article>

                <footer className="sticky-actions">
                  <button className="secondary-button" onClick={() => setScreen("detail")} type="button">
                    Back to coffee
                  </button>
                  <button className="primary-button" onClick={() => void submitReview()} type="button">
                    {submitting ? "Refreshing..." : "Get new recommendations"}
                  </button>
                </footer>
              </>
            ) : null}
          </section>
        ) : null}
      </section>
    </main>
  );
}

export default App;
