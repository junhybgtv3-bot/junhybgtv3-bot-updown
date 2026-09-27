"""FastAPI application for the Updown server."""

from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/health", response_class=Response)
def health() -> Response:
    return Response(content='{"ok": true}', media_type="application/json")
