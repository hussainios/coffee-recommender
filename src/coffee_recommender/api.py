from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi import Query

from .api_models import ProcessUrlRequest, SubmitReviewRequest
from .application import ApplicationService, create_application_service


def _translate_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal server error.")


def create_app(service: ApplicationService | None = None) -> FastAPI:
    app = FastAPI(title="Coffee Recommender API")
    application_service = service or create_application_service()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/catalogue/coffees")
    def list_catalogue_coffees():
        try:
            return application_service.list_catalogue_coffees()
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.get("/review-session")
    def get_review_session():
        try:
            return application_service.get_review_session()
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.delete("/review-session")
    def clear_review_session():
        try:
            return application_service.clear_review_session()
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.get("/reviewed-coffees/catalogue/{coffee_id}")
    def get_catalogue_reviewed_coffee(coffee_id: str):
        try:
            return application_service.get_catalogue_reviewed_coffee(coffee_id)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.post("/reviewed-coffees/from-url")
    def get_reviewed_coffee_from_url(request: ProcessUrlRequest):
        try:
            return application_service.get_reviewed_coffee_from_url(request.url)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.post("/reviews/submit")
    def submit_review_route(request: SubmitReviewRequest):
        try:
            return application_service.submit_review(request)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    @app.get("/review-session/landscape")
    def build_landscape(show_surface: bool = Query(default=True)):
        try:
            return application_service.build_landscape(show_surface=show_surface)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _translate_exception(exc) from exc

    return app


app = create_app()
