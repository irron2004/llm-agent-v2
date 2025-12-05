def test_search_basic(client):
    """Search API 기본 동작 테스트"""
    resp = client.get("/api/search", params={"q": "PM 점검"})

    assert resp.status_code == 200
    data = resp.json()

    # 응답 구조 검증
    assert "query" in data
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert "has_next" in data

    # 아이템 형태 검증
    for item in data["items"]:
        assert "rank" in item
        assert "id" in item
        assert "title" in item
        assert "snippet" in item
        assert "score_display" in item


def test_search_pagination(client):
    """Search API 페이지네이션 테스트"""
    resp = client.get("/api/search", params={"q": "test", "page": 1, "size": 2})

    assert resp.status_code == 200
    data = resp.json()

    assert data["page"] == 1
    assert data["size"] == 2
    assert len(data["items"]) <= 2
