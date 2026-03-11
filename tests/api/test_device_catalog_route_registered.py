from backend.api.main import create_app


def test_device_catalog_route_is_registered() -> None:
    app = create_app()
    route_paths = {route.path for route in app.routes}

    assert "/api/device-catalog" in route_paths
