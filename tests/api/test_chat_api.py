def test_chat_simple(client):
    payload = {
        "message": "PM 예방 점검 주기는?",
        "top_k": 3,
    }

    resp = client.post("/api/chat/simple", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == payload["message"]
    assert data["answer"].startswith("응답: ")
    assert "SIMPLE_PROMPT" in data["answer"]  # 시스템 프롬프트가 주입되었는지 확인
    assert data["retrieved_docs"] == []  # LLM-only
    assert len(data["follow_ups"]) > 0


def test_chat_retrieval_with_history(client):
    payload = {
        "message": "더 자세히 설명해줘",
        "history": [
            {"role": "user", "content": "PM이 뭐야?"},
            {"role": "assistant", "content": "PM은 예방 정비입니다."},
        ],
        "top_k": 2,
    }

    resp = client.post("/api/chat/retrieval", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == payload["message"]
    assert len(data["retrieved_docs"]) <= payload["top_k"]
    assert len(data["retrieved_docs"]) > 0
