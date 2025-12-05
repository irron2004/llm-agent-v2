def test_preprocessing_apply(client):
    payload = {"text": "  Hello   World  "}

    resp = client.post("/preprocessing/apply", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed_text"] == "Hello World"


def test_preprocessing_apply_with_level_override(client):
    payload = {"text": "  spaced   out  ", "level": "L5"}

    resp = client.post("/preprocessing/apply", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed_text"] == "spaced out"
