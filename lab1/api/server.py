import asyncio as aio
import contextlib as ctx
import datetime as dates
import json as js
import threading as th
import typing as ty

import fastapi as fa
import fastapi.responses as responses
import pydantic as pd
import api.agent as ag


@ctx.asynccontextmanager
async def lifespan(app):
    # Загружаем модель один раз на запуск сервера, не на каждый вопрос.
    app.state.agent = await aio.to_thread(ag.Agent)
    yield


app = fa.FastAPI(
    title="HR Agent — лабораторная 1",
    lifespan=lifespan,
)

CHAT_FOLDER = ag.FOLDER / "data" / "chats"
session_locks = {}


class Question(pd.BaseModel):
    model_config = pd.ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question: str = pd.Field(min_length=1, max_length=4000)
    stream: bool = False


class Source(pd.BaseModel):
    source: str
    section: str


class Answer(pd.BaseModel):
    answer: str
    sources: list[Source] = pd.Field(default_factory=list)
    tools_used: list[str] = pd.Field(default_factory=list)
    session_id: str | None = None


class Uploaded(pd.BaseModel):
    source: str
    chunks_added: int


def chat_events(session_id, question):
    lock = session_locks.setdefault(session_id, th.Lock())

    with lock:
        path = CHAT_FOLDER / (session_id + ".json")
        history = (
            js.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else []
        )

        for event, data in app.state.agent.chat_events(question, history):
            if event == "done":
                CHAT_FOLDER.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    js.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(path)
                data["session_id"] = session_id

            yield event, data


def answer_response(events, stream):
    if stream:

        def sse():
            try:
                for event, data in events:
                    yield (
                        f"event: {event}\n"
                        f"data: {js.dumps(data, ensure_ascii=False)}\n\n"
                    )
            except Exception as error:
                data = {"detail": str(error)}
                yield (
                    f"event: error\n"
                    f"data: {js.dumps(data, ensure_ascii=False)}\n\n"
                )
            finally:
                events.close()

        return responses.StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        for event, data in events:
            if event == "done":
                return Answer(**data)
        raise RuntimeError("Не получен итоговый ответ")
    except Exception as error:
        raise fa.HTTPException(
            status_code=502,
            detail=str(error),
        ) from error
    finally:
        events.close()


SSE_RESPONSE = {200: {"content": {"text/event-stream": {}}}}


@app.post(
    "/api/ask",
    response_model=Answer,
    responses=SSE_RESPONSE,
)
def ask(body: Question):
    """Разовый запрос к агенту с инструментами, без сохранения истории."""
    return answer_response(
        app.state.agent.ask_events(body.question),
        body.stream,
    )


@app.post(
    "/api/chat/{session_id}",
    response_model=Answer,
    responses=SSE_RESPONSE,
)
def chat(
    body: Question,
    session_id: str = fa.Path(pattern=r"^[A-Za-z0-9_-]{1,80}$"),
):
    """Агент с инструментами. История восстанавливается из JSON этой сессии."""
    return answer_response(
        chat_events(session_id, body.question),
        body.stream,
    )


@app.post("/api/upload", response_model=Uploaded)
def upload(
    file: fa.UploadFile = fa.File(...),
    doc_type: str = fa.Form(..., min_length=1, max_length=80),
    date: dates.date = fa.Form(...),
    department: ty.Literal[
        "All",
        "Engineering",
        "Sales",
        "Support",
        "Finance",
        "HR",
        "Operations",
    ] = fa.Form("All"),
):
    """Загрузка нового документа; date — дата выпуска, department — отдел применения."""
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    if not filename or len(filename) > 150:
        raise fa.HTTPException(
            status_code=400,
            detail="Нужно имя файла длиной до 150 символов",
        )

    if doc_type not in ag.DOC_TYPES:
        raise fa.HTTPException(
            status_code=400,
            detail="Неизвестный doc_type; для своего документа используй other",
        )

    try:
        content = file.file.read(10 * 1024 * 1024 + 1)
    finally:
        file.file.close()

    if len(content) > 10 * 1024 * 1024:
        raise fa.HTTPException(
            status_code=413,
            detail="Максимальный размер файла — 10 МБ",
        )

    try:
        return app.state.agent.upload(
            filename,
            content,
            doc_type,
            date.isoformat(),
            department,
        )
    except FileExistsError as error:
        raise fa.HTTPException(
            status_code=409,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise fa.HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
