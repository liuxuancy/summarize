import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Doc Summarise")


class SummariseRequest(BaseModel):
    text: str


def get_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY 未设置，请在 .env 中填写")
    return OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/summarise")
def summarise(req: SummariseRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="context is empty")

    client = get_client()
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-1-flash-260910",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes concise, well-structured summaries. "
                    "Reply in the same language as the document.",
                },
                {"role": "user", "content": f"please summary the input context,just send the summarised text back and do not have title：\n\n{text}"},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to call agent: {e}")

    return {"summary": resp.choices[0].message.content or ""}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
