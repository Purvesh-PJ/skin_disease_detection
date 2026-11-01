"""
Integration Tests: Health & Base Routing
----------------------------------------
Tests root status, uptime, health checks, and global error handling.
"""


class TestHealthAndSystemRoutes:
    """Tests system endpoints and HTTP error handlers."""

    def test_health_check_returns_online(self, client):
        """GET / should return 200 with online status and service metadata."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.get_json()
        assert data is not None
        assert data["status"] == "online"
        assert "Skin Disease Detection" in data["service"]
        assert "version" in data

    def test_non_existent_route_returns_404(self, client):
        """Requesting an undefined endpoint should return a structured 404 response."""
        response = client.get("/api/v1/does_not_exist")
        assert response.status_code == 404

        data = response.get_json()
        assert data is not None
        assert data["status"] == "error"

    def test_method_not_allowed_on_get_predict(self, client):
        """GET request on /predict (which only accepts POST) should return 405."""
        response = client.get("/predict")
        assert response.status_code == 405

        data = response.get_json()
        assert data is not None
        assert data["status"] == "error"

