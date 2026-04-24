from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv_core import create_app

from submission.env.models import FarmAction, FarmObservation
from submission.env.env import PrintFarmEnvironment

app = create_app(
    PrintFarmEnvironment,
    FarmAction,
    FarmObservation,
    env_name="printfarm-env",
    max_concurrent_envs=1,
)


async def _proxy_to_root(path: str, request: Request) -> JSONResponse:
    body = await request.body()
    from starlette.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        f"/{path}",
        content=body,
        headers={"content-type": "application/json"},
    )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/env/reset")
async def env_reset_alias(request: Request):
    return await _proxy_to_root("reset", request)


@app.post("/env/step")
async def env_step_alias(request: Request):
    return await _proxy_to_root("step", request)


@app.post("/env/create")
async def env_create_alias(request: Request):
    return await _proxy_to_root("create", request)


@app.get("/env/state")
async def env_state_alias(request: Request):
    from starlette.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/state")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>PrintFarmEnv v2 — Submission Server</h1><p>Running</p>"
