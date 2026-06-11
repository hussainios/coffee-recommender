from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .api_models import (
    CatalogueCoffeeSummary,
    LandscapeRequest,
    LandscapeResponse,
    ProcessUrlRequest,
    ProcessUrlResponse,
    ReviewedCoffeePayload,
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

    def get_catalogue_coffee(self, coffee_id: str) -> ReviewedCoffeePayload:
        payload = self._request("GET", f"/catalogue/coffees/{quote(coffee_id)}")
        return ReviewedCoffeePayload.model_validate(payload)

    def process_reviewed_coffee_url(self, url: str) -> ProcessUrlResponse:
        payload = self._request(
            "POST",
            "/reviewed-coffee/url",
            ProcessUrlRequest(url=url).model_dump(mode="json"),
        )
        return ProcessUrlResponse.model_validate(payload)

    def submit_review(self, request: SubmitReviewRequest) -> SubmitReviewResponse:
        payload = self._request(
            "POST",
            "/reviews/submit",
            request.model_dump(mode="json"),
        )
        return SubmitReviewResponse.model_validate(payload)

    def build_landscape(self, request: LandscapeRequest) -> LandscapeResponse:
        payload = self._request(
            "POST",
            "/landscape",
            request.model_dump(mode="json"),
        )
        return LandscapeResponse.model_validate(payload)
