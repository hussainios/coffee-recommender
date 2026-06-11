from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .api_models import (
    CatalogueCoffeeSummary,
    LandscapeResponse,
    ProcessUrlRequest,
    ReviewedCoffeeDetails,
    ReviewSessionPayload,
    SubmitReviewRequest,
    SubmitReviewResponse,
)
from .config import get_api_base_url


class ApiClientError(RuntimeError):
    pass


class StreamlitApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or get_api_base_url()).rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.reason
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = payload.get("detail", detail)
            except json.JSONDecodeError:
                pass
            raise ApiClientError(str(detail)) from exc
        except URLError as exc:
            raise ApiClientError(
                "Could not reach the FastAPI backend. Start it with: "
                "uvicorn coffee_recommender.api:app --reload"
            ) from exc

        if not raw:
            return None
        return json.loads(raw)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_catalogue_coffees(self) -> list[CatalogueCoffeeSummary]:
        payload = self._request("GET", "/catalogue/coffees")
        return [CatalogueCoffeeSummary.model_validate(item) for item in payload]

    def get_review_session(self) -> ReviewSessionPayload:
        payload = self._request("GET", "/review-session")
        return ReviewSessionPayload.model_validate(payload)

    def clear_review_session(self) -> ReviewSessionPayload:
        payload = self._request("DELETE", "/review-session")
        return ReviewSessionPayload.model_validate(payload)

    def get_catalogue_reviewed_coffee(self, coffee_id: str) -> ReviewedCoffeeDetails:
        payload = self._request("GET", f"/reviewed-coffees/catalogue/{quote(coffee_id)}")
        return ReviewedCoffeeDetails.model_validate(payload)

    def get_reviewed_coffee_from_url(self, url: str) -> ReviewedCoffeeDetails:
        payload = self._request(
            "POST",
            "/reviewed-coffees/from-url",
            ProcessUrlRequest(url=url).model_dump(mode="json"),
        )
        return ReviewedCoffeeDetails.model_validate(payload)

    def submit_review(self, request: SubmitReviewRequest) -> SubmitReviewResponse:
        payload = self._request(
            "POST",
            "/reviews/submit",
            request.model_dump(mode="json"),
        )
        return SubmitReviewResponse.model_validate(payload)

    def build_landscape(self, show_surface: bool = True) -> LandscapeResponse:
        query = urlencode({"show_surface": str(show_surface).lower()})
        payload = self._request("GET", f"/review-session/landscape?{query}")
        return LandscapeResponse.model_validate(payload)
