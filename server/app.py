from openenv_core import create_app
from fastapi.responses import HTMLResponse
from printfarm_env.models import FarmAction, FarmObservation
from printfarm_env.env import PrintFarmEnvironment

app = create_app(
    PrintFarmEnvironment,
    FarmAction,
    FarmObservation,
    env_name="printfarm-env",
    max_concurrent_envs=1,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PrintFarmEnv</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .card {
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 16px;
                padding: 48px;
                max-width: 520px;
                text-align: center;
                backdrop-filter: blur(12px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            }
            .emoji { font-size: 64px; margin-bottom: 16px; }
            h1 {
                font-size: 28px;
                font-weight: 700;
                background: linear-gradient(90deg, #38bdf8, #34d399);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 12px;
            }
            p { color: #94a3b8; line-height: 1.6; margin-bottom: 24px; }
            .badge {
                display: inline-block;
                background: rgba(52, 211, 153, 0.15);
                color: #34d399;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }
            .endpoints {
                margin-top: 32px;
                text-align: left;
                background: rgba(15, 23, 42, 0.6);
                border-radius: 10px;
                padding: 20px;
            }
            .endpoints h3 {
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #64748b;
                margin-bottom: 12px;
            }
            .endpoint {
                font-family: 'SF Mono', 'Fira Code', monospace;
                font-size: 13px;
                color: #38bdf8;
                padding: 4px 0;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="emoji">🖨️</div>
            <h1>PrintFarmEnv</h1>
            <p>A hardware DevOps environment for managing a fleet of 3D printers and inventory.</p>
            <span class="badge">✓ Running</span>
            <div class="endpoints">
                <h3>API Endpoints</h3>
                <div class="endpoint">GET  /health</div>
                <div class="endpoint">GET  /docs</div>
                <div class="endpoint">POST /env/create</div>
                <div class="endpoint">POST /env/step</div>
                <div class="endpoint">POST /env/reset</div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "PrintFarmEnv"}
