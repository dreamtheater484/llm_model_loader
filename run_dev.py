from backend.app.config import DEFAULT_HOST, DEFAULT_PORT

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=True)

