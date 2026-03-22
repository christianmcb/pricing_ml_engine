from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import load_config
from src.data_processing import validate_inference_dataframe
from src.model_registry import get_model_metadata, get_model_path


config = load_config()
selected_model_path = get_model_path(config["artifacts"]["model_path"])
selected_model_metadata = get_model_metadata(
    config["artifacts"]["model_path"],
    selected_model_path,
)
model = joblib.load(selected_model_path)

app = FastAPI(title="Pricing ML Engine API")

LANDING_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Pricing ML Engine API</title>
    <style>
        :root {
            --bg: #f2f4ef;
            --ink: #1a231f;
            --muted: #5f6c64;
            --card: #ffffff;
            --line: #d8e0d9;
            --brand: #1f6f46;
            --brand-dark: #174e33;
            --chip: #e6efe8;
            --code: #f8faf7;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
            background:
                radial-gradient(1200px 500px at 100% -10%, #dde9df 0%, transparent 70%),
                linear-gradient(180deg, #f7faf6 0%, var(--bg) 100%);
        }
        .container {
            max-width: 1080px;
            margin: 30px auto 60px;
            padding: 0 18px;
        }
        .hero {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 16px 36px rgba(21, 34, 27, 0.08);
            animation: enter 420ms ease-out;
        }
        @keyframes enter {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .title {
            margin: 0 0 8px;
            font-size: 34px;
            line-height: 1.15;
            letter-spacing: 0.2px;
        }
        .subtitle {
            margin: 0;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.45;
        }
        .status-row {
            margin-top: 14px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--chip);
            font-size: 13px;
            color: #214330;
        }
        .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #d19c2e;
        }
        .dot.ok { background: #2b9a5e; }
        .dot.bad { background: #c54242; }

        .links {
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .link-btn {
            text-decoration: none;
            color: #fff;
            background: linear-gradient(120deg, var(--brand), var(--brand-dark));
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 14px;
        }
        .link-btn.secondary {
            color: #1f3d2d;
            background: #edf3ee;
            border: 1px solid var(--line);
        }

        .section {
            margin-top: 18px;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
        }
        .section h2 {
            margin: 0 0 10px;
            font-size: 22px;
        }
        .section p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }
        .endpoint-grid, .form-grid {
            margin-top: 12px;
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        }
        .card {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
            background: #fcfdfb;
        }
        .method {
            font-family: "Courier New", monospace;
            font-size: 12px;
            color: #fff;
            background: #2b7a4f;
            border-radius: 6px;
            padding: 2px 6px;
            margin-right: 8px;
        }
        .path {
            font-family: "Courier New", monospace;
            color: #294836;
            font-size: 14px;
        }
        .code {
            position: relative;
            margin-top: 10px;
        }
        pre {
            margin: 0;
            border: 1px solid var(--line);
            background: var(--code);
            border-radius: 10px;
            padding: 12px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.45;
        }
        .copy {
            position: absolute;
            top: 8px;
            right: 8px;
            border: 1px solid var(--line);
            background: #fff;
            color: #1f3d2d;
            border-radius: 8px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            color: #4f6056;
            font-size: 13px;
        }
        input, select {
            width: 100%;
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 9px;
            padding: 9px 10px;
            font-size: 14px;
        }
        .actions {
            margin-top: 12px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            border: none;
            cursor: pointer;
            border-radius: 9px;
            padding: 10px 14px;
            font-size: 14px;
        }
        .btn-primary {
            color: #fff;
            background: linear-gradient(120deg, var(--brand), var(--brand-dark));
        }
        .btn-subtle {
            border: 1px solid var(--line);
            background: #f2f6f2;
            color: #254332;
        }
        .results {
            display: grid;
            gap: 12px;
            grid-template-columns: 1fr 1fr;
            margin-top: 12px;
        }
        .footnote {
            margin-top: 12px;
            font-size: 12px;
            color: #6b786f;
        }
        @media (max-width: 780px) {
            .title { font-size: 28px; }
            .results { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="container">
        <section class="hero">
            <h1 class="title">Pricing ML Engine API</h1>
            <p class="subtitle">
                Production-style vehicle insurance conversion scoring API with dynamic premium recommendation.
                Use this page as a quick test console and portfolio-friendly technical overview.
            </p>
            <div class="status-row">
                <span class="chip"><span id="api-dot" class="dot"></span>API status: <strong id="api-status">checking...</strong></span>
                <span class="chip">Model source: <code>models/current/model.joblib</code></span>
                <span class="chip">Primary endpoint: <code>POST /quote</code></span>
            </div>
            <div class="links">
                <a class="link-btn" href="/docs" target="_blank" rel="noreferrer">Open Swagger Docs</a>
                <a class="link-btn secondary" href="/redoc" target="_blank" rel="noreferrer">Open ReDoc</a>
                <a class="link-btn secondary" href="/health" target="_blank" rel="noreferrer">Health JSON</a>
            </div>
        </section>

        <section class="section">
            <h2>Endpoints</h2>
            <div class="endpoint-grid">
                <article class="card">
                    <div><span class="method">GET</span><span class="path">/health</span></div>
                    <p>Operational status check used for deployment probes.</p>
                </article>
                <article class="card">
                    <div><span class="method">POST</span><span class="path">/quote</span></div>
                    <p>Returns conversion probability and recommended premium.</p>
                </article>
                <article class="card">
                    <div><span class="method">GET</span><span class="path">/docs</span></div>
                    <p>OpenAPI docs for schema-first integration.</p>
                </article>
            </div>
        </section>

        <section class="section">
            <h2>Quickstart</h2>
            <p>Use this cURL to test the API from terminal.</p>
            <div class="code">
                <button class="copy" data-copy-target="curl-snippet">Copy</button>
                <pre id="curl-snippet">curl -X POST http://127.0.0.1:8000/quote \\
  -H "Content-Type: application/json" \\
  -d '{
    "Gender": "Male",
    "Age": 35,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Vehicle_Age": "1-2 Year",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 30000,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 120
  }'</pre>
            </div>
        </section>

        <section class="section">
            <h2>Interactive Playground</h2>
            <p>Submit a request and inspect the raw response JSON in real time.</p>
            <form id="quote-form">
                <div class="form-grid">
                    <div><label>Gender</label><select name="Gender"><option>Male</option><option>Female</option></select></div>
                    <div><label>Age</label><input name="Age" type="number" value="35" min="18" max="120" required /></div>
                    <div><label>Driving License</label><select name="Driving_License"><option value="1">1</option><option value="0">0</option></select></div>
                    <div><label>Region Code</label><input name="Region_Code" type="number" step="0.1" value="28.0" required /></div>
                    <div><label>Previously Insured</label><select name="Previously_Insured"><option value="0">0</option><option value="1">1</option></select></div>
                    <div><label>Vehicle Age</label><select name="Vehicle_Age"><option>&lt; 1 Year</option><option>1-2 Year</option><option>&gt; 2 Years</option></select></div>
                    <div><label>Vehicle Damage</label><select name="Vehicle_Damage"><option>Yes</option><option>No</option></select></div>
                    <div><label>Annual Premium</label><input name="Annual_Premium" type="number" step="0.01" min="0" value="30000" required /></div>
                    <div><label>Policy Sales Channel</label><input name="Policy_Sales_Channel" type="number" step="0.1" value="26.0" required /></div>
                    <div><label>Vintage</label><input name="Vintage" type="number" min="0" value="120" required /></div>
                </div>
                <div class="actions">
                    <button class="btn-primary" type="submit">Run Quote</button>
                    <button class="btn-subtle" type="button" id="preset-low">Load Low-Conversion Preset</button>
                    <button class="btn-subtle" type="button" id="preset-high">Load High-Conversion Preset</button>
                </div>
            </form>
            <div class="results">
                <div>
                    <p><strong>Request Payload</strong></p>
                    <pre id="request-json">Waiting...</pre>
                </div>
                <div>
                    <p><strong>Response</strong></p>
                    <pre id="response-json">Waiting...</pre>
                </div>
            </div>
            <p class="footnote">Tip: this page is meant for recruiter/demo readability; developers can still use OpenAPI docs for full schema details.</p>
        </section>
    </main>

    <script>
        const healthDot = document.getElementById("api-dot");
        const healthText = document.getElementById("api-status");

        async function checkHealth() {
            try {
                const r = await fetch("/health");
                if (!r.ok) {
                    throw new Error("status " + r.status);
                }
                healthDot.classList.add("ok");
                healthText.textContent = "online";
            } catch (_err) {
                healthDot.classList.add("bad");
                healthText.textContent = "offline";
            }
        }

        function toPayload(form) {
            const fd = new FormData(form);
            return {
                Gender: fd.get("Gender"),
                Age: Number(fd.get("Age")),
                Driving_License: Number(fd.get("Driving_License")),
                Region_Code: Number(fd.get("Region_Code")),
                Previously_Insured: Number(fd.get("Previously_Insured")),
                Vehicle_Age: fd.get("Vehicle_Age"),
                Vehicle_Damage: fd.get("Vehicle_Damage"),
                Annual_Premium: Number(fd.get("Annual_Premium")),
                Policy_Sales_Channel: Number(fd.get("Policy_Sales_Channel")),
                Vintage: Number(fd.get("Vintage"))
            };
        }

        const form = document.getElementById("quote-form");
        const reqEl = document.getElementById("request-json");
        const resEl = document.getElementById("response-json");

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const payload = toPayload(form);
            reqEl.textContent = JSON.stringify(payload, null, 2);
            resEl.textContent = "Requesting...";

            try {
                const r = await fetch("/quote", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const body = await r.json();
                resEl.textContent = JSON.stringify(body, null, 2);
            } catch (err) {
                resEl.textContent = "Request failed: " + err;
            }
        });

        document.getElementById("preset-low").addEventListener("click", () => {
            form.Gender.value = "Female";
            form.Age.value = 23;
            form.Driving_License.value = 1;
            form.Region_Code.value = 12.0;
            form.Previously_Insured.value = 1;
            form.Vehicle_Age.value = "< 1 Year";
            form.Vehicle_Damage.value = "No";
            form.Annual_Premium.value = 18000;
            form.Policy_Sales_Channel.value = 152.0;
            form.Vintage.value = 30;
        });

        document.getElementById("preset-high").addEventListener("click", () => {
            form.Gender.value = "Male";
            form.Age.value = 46;
            form.Driving_License.value = 1;
            form.Region_Code.value = 28.0;
            form.Previously_Insured.value = 0;
            form.Vehicle_Age.value = "> 2 Years";
            form.Vehicle_Damage.value = "Yes";
            form.Annual_Premium.value = 42000;
            form.Policy_Sales_Channel.value = 26.0;
            form.Vintage.value = 230;
        });

        document.querySelectorAll(".copy").forEach((button) => {
            button.addEventListener("click", async () => {
                const target = document.getElementById(button.dataset.copyTarget);
                try {
                    await navigator.clipboard.writeText(target.textContent);
                    const original = button.textContent;
                    button.textContent = "Copied";
                    setTimeout(() => { button.textContent = original; }, 1000);
                } catch (_err) {
                    button.textContent = "Copy failed";
                }
            });
        });

        checkHealth();
    </script>
</body>
</html>
"""


class PricingRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=120)
    Driving_License: int = Field(..., ge=0, le=1)
    Region_Code: float
    Previously_Insured: int = Field(..., ge=0, le=1)
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(..., ge=0)
    Policy_Sales_Channel: float
    Vintage: int = Field(..., ge=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(selected_model_path),
        "run_id": selected_model_metadata.get("run_id"),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_path": str(selected_model_path),
        "metadata": selected_model_metadata,
    }


@app.get("/", response_class=HTMLResponse)
def landing_page():
    """Returns a simple interactive UI for testing the quote endpoint."""
    return LANDING_PAGE_HTML


@app.post("/quote")
def quote(request: PricingRequest):
    """Scores a pricing request and returns the recommended premium and price segment."""
    input_df = pd.DataFrame([request.model_dump()])
    input_df = validate_inference_dataframe(input_df)

    conversion_probability = float(model.predict_proba(input_df)[:, 1][0])
    predicted_conversion = int(model.predict(input_df)[0])

    base_premium = config["pricing"]["base_premium"]
    demand_multiplier = config["pricing"]["demand_multiplier"]

    demand_adjustment = conversion_probability * demand_multiplier
    recommended_premium = round(base_premium + demand_adjustment, 2)

    if conversion_probability >= 0.75:
        price_segment = "high-conversion"
    elif conversion_probability >= 0.40:
        price_segment = "mid-conversion"
    else:
        price_segment = "low-conversion"

    return {
        "conversion_probability": round(conversion_probability, 4),
        "predicted_conversion": predicted_conversion,
        "base_premium": base_premium,
        "demand_adjustment": round(demand_adjustment, 2),
        "recommended_premium": recommended_premium,
        "price_segment": price_segment,
    }
