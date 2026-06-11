from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

from .coffee_dimensions import PROCESS_DIMENSIONS, SENSORY_DIMENSIONS
from .landscape import (
    CoffeeFeatures,
    LandscapeConfig,
    ReviewEvent,
    estimate_temperature,
    score_candidate,
)


def _feature_vector(coffee: CoffeeFeatures) -> list[float]:
    return [
        *(float(coffee.sensory.get(dimension, 0.0)) for dimension in SENSORY_DIMENSIONS),
        *(float(coffee.process.get(dimension, 0.0)) for dimension in PROCESS_DIMENSIONS),
        *coffee.embedding,
    ]


def _format_hover(coffee: CoffeeFeatures, score: float, role: str) -> str:
    return (
        f"<b>{coffee.name}</b><br>"
        f"ID: {coffee.coffee_id}<br>"
        f"Role: {role}<br>"
        f"Score: {score:.4f}"
    )


def _project_features(features: dict[str, CoffeeFeatures]) -> dict[str, tuple[float, float]]:
    coffee_ids = list(features)
    matrix = np.array([_feature_vector(features[coffee_id]) for coffee_id in coffee_ids], dtype=float)
    projection = PCA(n_components=2, random_state=0).fit_transform(matrix)
    return {
        coffee_id: (float(point[0]), float(point[1]))
        for coffee_id, point in zip(coffee_ids, projection)
    }


def _score_features(
    scoring_features: dict[str, CoffeeFeatures],
    reviews: list[ReviewEvent],
    config: LandscapeConfig = LandscapeConfig(),
) -> dict[str, tuple[float, dict[str, Any]]]:
    temperature = estimate_temperature(
        scoring_features,
        weights=config.distance_weights,
        neighbor_rank=config.neighbor_rank,
        target_kernel=config.target_kernel_at_neighbor,
        default_temperature=config.default_temperature,
    )
    return {
        coffee_id: score_candidate(
            coffee,
            reviews,
            scoring_features,
            temperature=temperature,
            weights=config.distance_weights,
        )
        for coffee_id, coffee in scoring_features.items()
    }


def _surface_grid(
    coffee_ids: list[str],
    coordinates: dict[str, tuple[float, float]],
    scored: dict[str, tuple[float, dict[str, Any]]],
    grid_size: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(coffee_ids) < 4:
        return None

    points = np.array([coordinates[coffee_id] for coffee_id in coffee_ids], dtype=float)
    values = np.array([scored[coffee_id][0] for coffee_id in coffee_ids], dtype=float)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    if x_min == x_max or y_min == y_max:
        return None

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
    if np.isnan(grid_z).all():
        return None

    return grid_x, grid_y, grid_z


def build_projected_score_landscape_figure(
    catalogue_features: dict[str, CoffeeFeatures],
    scoring_features: dict[str, CoffeeFeatures],
    reviews: list[ReviewEvent],
    top_recommendations: list[dict[str, Any]],
    show_surface: bool = False,
) -> go.Figure | None:
    if not reviews or len(scoring_features) < 3:
        return None

    coordinates = _project_features(scoring_features)
    scored = _score_features(scoring_features, reviews)
    reviewed_ids = {str(review.get("coffee_id")) for review in reviews}
    top_ids = {str(item.get("coffee_id")) for item in top_recommendations}

    catalogue_ids = [
        coffee_id
        for coffee_id in catalogue_features
        if coffee_id in coordinates and coffee_id not in reviewed_ids and coffee_id not in top_ids
    ]
    top_catalogue_ids = [
        coffee_id
        for coffee_id in catalogue_features
        if coffee_id in coordinates and coffee_id in top_ids and coffee_id not in reviewed_ids
    ]
    reviewed_catalogue_ids = [
        coffee_id
        for coffee_id in catalogue_features
        if coffee_id in coordinates and coffee_id in reviewed_ids
    ]
    temporary_reviewed_ids = [
        coffee_id
        for coffee_id in scoring_features
        if coffee_id not in catalogue_features and coffee_id in reviewed_ids and coffee_id in coordinates
    ]

    all_scores = [score for score, _ in scored.values()]
    color_min = min(all_scores) if all_scores else 0.0
    color_max = max(all_scores) if all_scores else 1.0

    fig = go.Figure()

    if show_surface:
        surface_ids = [
            coffee_id
            for coffee_id in scoring_features
            if coffee_id in coordinates and coffee_id in scored
        ]
        surface = _surface_grid(surface_ids, coordinates, scored)
        if surface is not None:
            grid_x, grid_y, grid_z = surface
            fig.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=grid_z,
                    name="Interpolated surface",
                    colorscale="Viridis",
                    cmin=color_min,
                    cmax=color_max,
                    opacity=0.48,
                    showscale=False,
                    hovertemplate=(
                        "PCA 1: %{x:.2f}<br>"
                        "PCA 2: %{y:.2f}<br>"
                        "Interpolated score: %{z:.4f}<extra></extra>"
                    ),
                )
            )

    def add_trace(coffee_ids: list[str], name: str, role: str, size: int, symbol: str) -> None:
        if not coffee_ids:
            return
        scores = [scored[coffee_id][0] for coffee_id in coffee_ids]
        fig.add_trace(
            go.Scatter3d(
                x=[coordinates[coffee_id][0] for coffee_id in coffee_ids],
                y=[coordinates[coffee_id][1] for coffee_id in coffee_ids],
                z=scores,
                mode="markers",
                name=name,
                text=[
                    _format_hover(scoring_features[coffee_id], scored[coffee_id][0], role)
                    for coffee_id in coffee_ids
                ],
                hoverinfo="text",
                marker={
                    "size": size,
                    "symbol": symbol,
                    "color": scores,
                    "colorscale": "Viridis",
                    "cmin": color_min,
                    "cmax": color_max,
                    "opacity": 0.86,
                    "colorbar": {"title": "Score"} if name == "Catalogue" else None,
                    "line": {"width": 1, "color": "#2f2f2f"},
                },
            )
        )

    add_trace(catalogue_ids, "Catalogue", "catalogue candidate", 5, "circle")
    add_trace(top_catalogue_ids, "Top recommendations", "top recommendation", 8, "diamond")
    add_trace(reviewed_catalogue_ids, "Reviewed catalogue coffees", "reviewed", 9, "cross")
    add_trace(temporary_reviewed_ids, "Temporary reviewed coffees", "temporary reviewed", 9, "x")

    fig.update_layout(
        height=680,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        scene={
            "xaxis_title": "PCA 1",
            "yaxis_title": "PCA 2",
            "zaxis_title": "Score",
            "camera": {"eye": {"x": 1.5, "y": 1.7, "z": 1.15}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig
