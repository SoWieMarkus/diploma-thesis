"""Tests of the REST API against a database with synthetic data."""

from fastapi.testclient import TestClient

from gaze_estimator.api.app import create_app


def test_api_lists_and_labels(synthetic_db):
    client = TestClient(create_app(synthetic_db, web_dist=None))
    assert client.get("/api/models").json() == {
        "faces": ["dlib", "mediapipe"],
        "gazes": ["bad-model", "good-model"],
    }
    videos = client.get("/api/videos").json()
    assert [v["name"] for v in videos] == ["video0", "video1", "video2"]
    assert videos[0]["processed_frames"] == 80

    frames = client.get(f"/api/videos/{videos[0]['id']}/frames").json()
    frame_id = frames[0]["id"]
    detail = client.get(f"/api/frames/{frame_id}").json()
    assert {f["detector"] for f in detail["faces"]} == {"dlib", "mediapipe"}

    response = client.patch(f"/api/frames/{frame_id}", json={"label_looking": 2})
    assert response.json()["label_looking"] == 2
    # Fields that are not sent stay untouched, explicit null clears.
    assert (
        client.patch(f"/api/frames/{frame_id}", json={"label_blink": True}).json()["label_looking"]
        == 2
    )
    assert (
        client.patch(f"/api/frames/{frame_id}", json={"label_looking": None}).json()[
            "label_looking"
        ]
        is None
    )
    assert client.patch(f"/api/frames/{frame_id}", json={"label_looking": 5}).status_code == 422

    face = next(f for f in detail["faces"] if f["detected"])
    assert (
        client.patch(f"/api/faces/{face['id']}", json={"label_quality": 1}).json()["label_quality"]
        == 1
    )

    track = client.get(
        f"/api/videos/{videos[0]['id']}/gazes", params={"face": "mediapipe", "gaze": "good-model"}
    )
    assert len(track.json()) > 70
