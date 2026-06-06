def test_health_open(client):
    assert client.get("/up").json() == {"status": "ok"}


def test_models_require_token(client):
    assert client.get("/v1/steganography/models").status_code == 401
    assert client.get("/v1/steganalysis/models").status_code == 401


def test_catalogs(client, auth):
    steg = client.get("/v1/steganography/models", headers=auth)
    ana = client.get("/v1/steganalysis/models", headers=auth)
    assert steg.status_code == 200 and len(steg.json()) == 10
    assert ana.status_code == 200 and len(ana.json()) == 10


def test_encode_rejects_unknown_model(client, auth):
    r = client.post(
        "/v1/steganography/encode",
        headers=auth,
        json={"model_key": "bogus", "message": "x", "image_b64": "AAAA"},
    )
    assert r.status_code == 422
