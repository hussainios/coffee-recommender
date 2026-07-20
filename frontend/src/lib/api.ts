const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export type CatalogueCoffeeSummary = {
  coffee_id: string;
  name: string | null;
  roaster: string | null;
  origin_country: string | null;
  process: string | null;
  tasting_notes: string[];
};

export type ChangeRequestPayload = {
  direction: string;
  strength: number;
  target_value?: number | null;
  adjustment?: number | null;
};

export type AttributeOpinionPayload = {
  sentiment: string;
  strength: number;
};

export type ReviewEventPayload = {
  coffee_id: string;
  overall: number;
  change_requests: Record<string, ChangeRequestPayload>;
  attribute_opinions: Record<string, AttributeOpinionPayload>;
};

export type CoffeeFeaturesPayload = {
  coffee_id: string;
  name: string | null;
  sensory: Record<string, number>;
  process: Record<string, number>;
  embedding: number[];
};

export type RecommendationPayload = {
  coffee_id: string;
  name: string | null;
  score: number;
  temperature: number;
  roaster: string | null;
  origin_country: string | null;
  producer: string | null;
  process: string | null;
  tasting_notes: string[];
  source_url: string | null;
  debug: Record<string, unknown>;
};

export type ReviewSessionPayload = {
  review_events: ReviewEventPayload[];
  reviewed_feature_overrides: Record<string, CoffeeFeaturesPayload>;
  last_event: ReviewEventPayload | null;
  last_recommendations: RecommendationPayload[];
};

export type ReviewedCoffeeDetails = {
  features: CoffeeFeaturesPayload;
  metadata: Record<string, unknown> | null;
  sensory: Record<string, unknown> | null;
  source_type: "catalogue" | "external_url";
  normalized_url: string | null;
};

export type SubmitReviewRequest = {
  review_text: string;
  reviewed_coffee: ReviewedCoffeeDetails;
  top_k: number;
};

export type SubmitReviewResponse = {
  event: ReviewEventPayload;
  review_session: ReviewSessionPayload;
  recommendations: RecommendationPayload[];
};

export type CoffeeDetailPayload = {
  coffee_id: string;
  name: string | null;
  roaster: string | null;
  origin_country: string | null;
  region: string | null;
  producer: string | null;
  farm: string | null;
  process: string | null;
  roast_level: string | null;
  tasting_notes: string[];
  description: string | null;
  weight_g: number | null;
  price: number | null;
  currency: string | null;
  source_url: string | null;
  features: CoffeeFeaturesPayload;
};

const getApiBaseUrl = (): string =>
  (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail = "Request failed.";
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export const api = {
  listCatalogueCoffees(): Promise<CatalogueCoffeeSummary[]> {
    return request("/coffees");
  },

  getReviewSession(): Promise<ReviewSessionPayload> {
    return request("/review-session");
  },

  getCoffeeDetail(coffeeId: string): Promise<CoffeeDetailPayload> {
    return request(`/coffees/${encodeURIComponent(coffeeId)}`);
  },

  getCatalogueReviewedCoffee(coffeeId: string): Promise<ReviewedCoffeeDetails> {
    return request(`/reviewed-coffees/catalogue/${encodeURIComponent(coffeeId)}`);
  },

  submitReview(payload: SubmitReviewRequest): Promise<SubmitReviewResponse> {
    return request("/reviews", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
