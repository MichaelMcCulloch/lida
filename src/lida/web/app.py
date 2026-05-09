import asyncio
import gzip
import json
import os
import logging
import requests
import shutil
import tarfile
import time
import traceback
import multiprocessing
import matplotlib

matplotlib.use("Agg")  # Ensure headless execution
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
from fastapi import FastAPI, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from lida.utils import is_gzip_file, is_sqlite_file, is_tar_archive, read_dataframe, read_sqlite_tables

from ..components.litellm_generator import LiteLLMTextGenerator, token_sink
from ..datamodel import (
    GoalWebRequest,
    SummaryUrlRequest,
    TextGenerationConfig,
    VisualizeEditWebRequest,
    VisualizeEvalWebRequest,
    VisualizeExplainWebRequest,
    VisualizeRecommendRequest,
    VisualizeRepairWebRequest,
    VisualizeWebRequest,
    InfographicsRequest,
)
from ..components import Manager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lida")

textgen = LiteLLMTextGenerator()
logger.info("Initialized LiteLLMTextGenerator base_url=%s model=%s", textgen.base_url, textgen.model_name)

api_docs = os.environ.get("LIDA_API_DOCS", "False") == "True"


lida = Manager(text_gen=textgen)
app = FastAPI()
# allow cross origin requests for testing on localhost:800* ports only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api = FastAPI(root_path="/api/v1", docs_url="/docs" if api_docs else None, redoc_url=None)
app.mount("/api/v1", api)


root_file_path = os.path.dirname(os.path.abspath(__file__))
static_folder_root = os.path.join(root_file_path, "ui")
files_static_root = os.path.join(root_file_path, "files/")
data_folder = os.path.join(root_file_path, "files/data")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(files_static_root, exist_ok=True)
os.makedirs(static_folder_root, exist_ok=True)


# mount lida front end UI files
frontend_dist = os.path.join(root_file_path, "frontend/dist")
logger.info(f"Checking for frontend at: {frontend_dist}")
logger.info(f"Root path: {root_file_path}")
if os.path.exists(root_file_path):
    logger.info(f"Listing root path: {os.listdir(root_file_path)}")

if os.path.exists(frontend_dist):
    logger.info("New frontend found! Mounting...")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")
else:
    logger.info("New frontend NOT found. Falling back to legacy UI.")
    app.mount("/", StaticFiles(directory=static_folder_root, html=True), name="ui")
api.mount("/files", StaticFiles(directory=files_static_root, html=True), name="files")


# ---------------------------------------------------------------------------
# Pipeline events
#
# Stage names are intentionally a closed set so the frontend stage tracker
# can render a deterministic 5-step pipeline regardless of which dispatch
# branch (single-table, sqlite, or tar) the upload takes:
#
#   compress   - client-side gzip (frontend-only, server never emits)
#   upload     - bytes flying over the wire (frontend-only)
#   decompress - server unpacks .gz wrapper
#   analyze    - LLM-driven summary + heuristic goals + viz code generation
#   visualize  - parallel chart execution
# ---------------------------------------------------------------------------

EmitFn = Callable[[str, dict], None]


def _emit(emit: Optional[EmitFn], event: str, **payload: Any) -> None:
    if emit is None:
        return
    try:
        emit(event, payload)
    except Exception:
        logger.debug("emit callback raised; continuing", exc_info=True)


def execute_chart_worker(code, data_df, summary, library):
    """Worker function for executing charts in a separate process"""
    # Import locally to avoid pickling issues if global imports are complex,
    # though for top-level functions it's usually fine.
    # Re-importing matplotlib backend setting if needed for process safety
    import matplotlib

    matplotlib.use("Agg")  # Ensure non-interactive backend for worker
    from lida.infrastructure.adapters.executor import ChartExecutorAdapter

    executor = ChartExecutorAdapter()
    return executor.execute(code_specs=[code], data=data_df, summary=summary, library=library)


def summarize_and_visualize(
    file_location: str,
    display_name: str,
    textgen_config: TextGenerationConfig,
    run_visualize: bool,
    emit: Optional[EmitFn] = None,
    chart_workers_cap: Optional[int] = None,
    n_goals: int = 5,
) -> dict:
    """Run summarize -> goals -> charts for one tabular file.

    Two paths share the same summary stage:

    - **Streaming** (``emit`` provided): LLM goals are streamed and parsed
      incrementally; each closing ``{...}`` immediately submits an LLM viz
      task to a thread pool, so plot generation runs concurrently with goal
      generation. Charts execute in a shared process pool. Used by the
      /summarize/stream SSE endpoint.

    - **JSON** (``emit`` is None): legacy heuristic flow — instant goals via
      HeuristicGoalExplorer, instant viz codegen, parallel chart execution.
      Kept for the /summarize endpoint and CLI callers.
    """
    _emit(emit, "summary.started", file_name=display_name)

    summary_token_state = {"count": 0, "started_at": time.time()}

    def summary_sink(delta: str) -> None:
        summary_token_state["count"] += 1
        _emit(
            emit,
            "llm.token",
            file_name=display_name,
            phase="summary",
            delta=delta,
            tokens=summary_token_state["count"],
            elapsed_ms=int((time.time() - summary_token_state["started_at"]) * 1000),
        )

    with token_sink(summary_sink):
        summary = lida.summarize(
            data=file_location,
            file_name=display_name,
            summary_method="llm",
            textgen_config=textgen_config,
        )
    logger.info("pipeline[%s]: summary ready (%d tokens)", display_name, summary_token_state["count"])
    _emit(emit, "summary.ready", file_name=display_name, summary=jsonable_encoder(summary))
    result = {"summary": summary, "data_filename": display_name}

    if not run_visualize:
        return result

    if emit is not None:
        try:
            goals, charts = _llm_visualize_pipeline(
                summary,
                file_location,
                display_name,
                textgen_config,
                n_goals,
                emit,
                chart_workers_cap,
            )
            result["goals"] = goals
            result["charts"] = charts
        except Exception as exc:
            logger.error("pipeline[%s]: LLM viz pipeline failed: %s", display_name, exc, exc_info=True)
            _emit(emit, "table.error", file_name=display_name, error=str(exc))
        return result

    # JSON endpoint legacy path: heuristic goals + heuristic viz + parallel exec.
    try:
        goals = lida.goals(summary, n=20, textgen_config=textgen_config, method="heuristic")
        goals_with_code = []
        for i, goal in enumerate(goals):
            code_list = lida.visualize(
                summary=summary,
                goal=goal,
                textgen_config=textgen_config,
                library="seaborn",
                method="heuristic",
                return_error=True,
            )
            if code_list:
                goals_with_code.append((i, goal, code_list[0]))
        charts: list = []
        if goals_with_code:
            data_df = read_dataframe(file_location)
            cpu_budget = chart_workers_cap if chart_workers_cap else (os.cpu_count() or 4)
            max_workers = min(len(goals_with_code), max(1, cpu_budget))
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                future_to_index = {
                    executor.submit(execute_chart_worker, code, data_df, summary, "seaborn"): (i, goal)
                    for i, goal, code in goals_with_code
                }
                rendered = []
                for future in as_completed(future_to_index):
                    i, _ = future_to_index[future]
                    try:
                        chart_executed = future.result()
                        if chart_executed:
                            rendered.append((i, chart_executed[0]))
                    except Exception as exc:
                        logger.warning("Chart exception (idx %d, table=%s): %s", i, display_name, exc)
                rendered.sort(key=lambda x: x[0])
                charts = [r[1] for r in rendered]
        result["goals"] = goals
        result["charts"] = charts
    except Exception as exc:
        logger.error("Heuristic flow error for %s: %s", display_name, exc)
    return result


def _llm_visualize_pipeline(
    summary,
    file_location: str,
    display_name: str,
    textgen_config: TextGenerationConfig,
    n_goals: int,
    emit: EmitFn,
    chart_workers_cap: Optional[int],
) -> tuple:
    """Streaming LLM goals + pipelined parallel plot generation.

    The streaming token sink for the goals call routes deltas through
    _GoalStreamParser. Each closing ``{...}`` immediately submits a plot
    task to ``plot_pool`` (LLM viz call lives in this thread). Chart
    execution is offloaded to a shared ``chart_pool`` (matplotlib state
    is not thread-safe). Plot tasks run concurrently with later goals
    still being streamed by the model.
    """
    from lida.domain.models import Goal as DomainGoal

    parser = _GoalStreamParser()
    goals_streamed: list = []
    plot_futures: dict = {}

    cpu_budget = chart_workers_cap if chart_workers_cap else (os.cpu_count() or 4)
    chart_pool_size = max(1, min(n_goals, cpu_budget))
    plot_pool_size = max(2, min(n_goals, cpu_budget * 2))  # LLM is I/O-bound; over-thread is fine

    ctx = multiprocessing.get_context("spawn")
    chart_pool = ProcessPoolExecutor(max_workers=chart_pool_size, mp_context=ctx)
    plot_pool = ThreadPoolExecutor(
        max_workers=plot_pool_size,
        thread_name_prefix=f"lida-plot-{display_name[:8]}",
    )

    data_df = read_dataframe(file_location)

    goal_token_state = {"count": 0, "started_at": time.time()}

    def goal_sink(delta: str) -> None:
        goal_token_state["count"] += 1
        _emit(
            emit,
            "llm.token",
            file_name=display_name,
            phase="goals",
            delta=delta,
            tokens=goal_token_state["count"],
            elapsed_ms=int((time.time() - goal_token_state["started_at"]) * 1000),
        )
        for goal_dict in parser.feed(delta):
            idx = len(goals_streamed)
            try:
                goal = DomainGoal(
                    question=goal_dict.get("question", ""),
                    visualization=goal_dict.get("visualization", ""),
                    rationale=goal_dict.get("rationale", ""),
                    index=idx,
                )
            except Exception as exc:
                logger.warning("pipeline[%s]: streamed goal[%d] malformed: %s", display_name, idx, exc)
                continue
            goals_streamed.append(goal)
            _emit(
                emit,
                "goal.ready",
                file_name=display_name,
                index=idx,
                goal=jsonable_encoder(goal),
            )
            # Pipeline: kick off plot generation immediately, don't wait for
            # subsequent goals.
            future = plot_pool.submit(
                _run_plot_task,
                idx, goal, summary, data_df, display_name,
                textgen_config, emit, chart_pool,
            )
            plot_futures[idx] = future

    try:
        _emit(emit, "goals.started", file_name=display_name, n_requested=n_goals)
        try:
            with token_sink(goal_sink):
                final_goals = lida.goals(summary, n=n_goals, textgen_config=textgen_config)
        except Exception as exc:
            logger.error("pipeline[%s]: goal LLM call failed: %s", display_name, exc, exc_info=True)
            _emit(emit, "goals.error", file_name=display_name, error=str(exc))
            final_goals = goals_streamed
        logger.info(
            "pipeline[%s]: %d goals (%d via streaming, %d tokens)",
            display_name, len(final_goals), len(goals_streamed), goal_token_state["count"],
        )
        _emit(emit, "goals.ready", file_name=display_name, count=len(final_goals))

        # Edge case: adapter parse caught goals our streaming tracker missed.
        for i, goal in enumerate(final_goals):
            if i not in plot_futures:
                logger.info("pipeline[%s]: late-submitting plot[%d] missed by streamer", display_name, i)
                future = plot_pool.submit(
                    _run_plot_task,
                    i, goal, summary, data_df, display_name,
                    textgen_config, emit, chart_pool,
                )
                plot_futures[i] = future

        _emit(emit, "plots.started", file_name=display_name, total=len(plot_futures))
        chart_results: dict = {}
        for fut in as_completed(list(plot_futures.values())):
            try:
                idx, chart = fut.result()
                if chart is not None:
                    chart_results[idx] = chart
            except Exception as exc:
                logger.warning("pipeline[%s]: plot task crashed: %s", display_name, exc)

        ordered_charts = [chart_results[i] for i in sorted(chart_results.keys())]
        logger.info(
            "pipeline[%s]: %d/%d charts rendered",
            display_name, len(ordered_charts), len(plot_futures),
        )
        return final_goals, ordered_charts
    finally:
        plot_pool.shutdown(wait=True)
        chart_pool.shutdown(wait=True)


def _run_plot_task(
    goal_index: int,
    goal,
    summary,
    data_df,
    display_name: str,
    textgen_config: TextGenerationConfig,
    emit: EmitFn,
    chart_pool: ProcessPoolExecutor,
) -> tuple:
    """Per-goal worker: LLM viz code generation -> chart execution.

    Runs in a thread (LLM call is HTTP I/O so the GIL is friendly). Chart
    execution is dispatched to ``chart_pool`` because user-generated
    matplotlib code is not safe to run concurrently in threads.
    """
    _emit(
        emit,
        "plot.started",
        file_name=display_name,
        goal_index=goal_index,
        question=getattr(goal, "question", ""),
    )

    plot_token_state = {"count": 0, "started_at": time.time()}

    def plot_sink(delta: str) -> None:
        plot_token_state["count"] += 1
        _emit(
            emit,
            "llm.token",
            file_name=display_name,
            phase="plot",
            goal_index=goal_index,
            delta=delta,
            tokens=plot_token_state["count"],
            elapsed_ms=int((time.time() - plot_token_state["started_at"]) * 1000),
        )

    try:
        with token_sink(plot_sink):
            code_list = lida.visualize(
                summary=summary,
                goal=goal,
                textgen_config=textgen_config,
                library="seaborn",
                return_error=True,
            )
    except Exception as exc:
        logger.warning("plot[%s/%d]: LLM call failed: %s", display_name, goal_index, exc)
        _emit(emit, "plot.failed", file_name=display_name, goal_index=goal_index, error=str(exc))
        return goal_index, None

    if not code_list:
        _emit(emit, "plot.failed", file_name=display_name, goal_index=goal_index, error="no code generated")
        return goal_index, None

    code = code_list[0]
    _emit(emit, "plot.code.ready", file_name=display_name, goal_index=goal_index)

    try:
        future = chart_pool.submit(execute_chart_worker, code, data_df, summary, "seaborn")
        chart_executed = future.result()
    except Exception as exc:
        logger.warning("plot[%s/%d]: chart exec failed: %s", display_name, goal_index, exc)
        _emit(emit, "plot.failed", file_name=display_name, goal_index=goal_index, error=str(exc))
        return goal_index, None

    if not chart_executed:
        _emit(emit, "plot.failed", file_name=display_name, goal_index=goal_index, error="exec returned empty")
        return goal_index, None

    chart = chart_executed[0]
    _emit(
        emit,
        "chart.rendered",
        file_name=display_name,
        goal_index=goal_index,
        chart=jsonable_encoder(chart),
    )
    return goal_index, chart


@api.post("/visualize")
async def visualize_data(req: VisualizeWebRequest) -> dict:
    """Generate goals given a dataset summary"""
    print(
        f"DEBUG: Received visualize request for goal: {req.goal.question if req.goal else 'None'}",
        flush=True,
    )
    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()

        # Load data if library is not altair (which uses URL reference)
        data = None
        if req.library != "altair":
            file_path = os.path.join(data_folder, req.summary.file_name)
            if os.path.exists(file_path):
                data = read_dataframe(file_path)
            else:
                logger.warning(f"Data file not found at {file_path}")

        try:
            charts = lida.visualize(
                summary=req.summary,
                goal=req.goal,
                textgen_config=textgen_config,
                library=req.library,
                return_error=True,
                data=data,
            )
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            logger.error(traceback.format_exc())
            return {"status": False, "message": f"Error generating charts: {str(e)}"}
        logger.info(f"Found {len(charts)} charts for goal: {req.goal.question}")
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {
            "status": True,
            "charts": charts,
            "message": "Successfully generated charts.",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization goals: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error generating visualization goals. {str(exception_error)}",
        }


@api.post("/visualize/edit")
async def edit_visualization(req: VisualizeEditWebRequest) -> dict:
    """Given a visualization code, and a goal, generate a new visualization"""
    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.edit(
            code=req.code,
            summary=req.summary,
            instructions=req.instructions,
            textgen_config=textgen_config,
            library=req.library,
            return_error=True,
        )

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {
            "status": True,
            "charts": charts,
            "message": "Successfully edited charts.",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization edits: {str(exception_error)}")
        logger.debug(traceback.format_exc())
        return {"status": False, "message": "Error generating visualization edits."}


@api.post("/visualize/repair")
async def repair_visualization(req: VisualizeRepairWebRequest) -> dict:
    """Given a visualization goal and some feedback, generate a new visualization that addresses the feedback"""

    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.repair(
            code=req.code,
            feedback=req.feedback,
            goal=req.goal,
            summary=req.summary,
            textgen_config=textgen_config,
            library=req.library,
            return_error=True,
        )

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {
            "status": True,
            "charts": charts,
            "message": "Successfully generated chart repairs",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization repairs: {str(exception_error)}")
        return {"status": False, "message": "Error generating visualization repairs."}


@api.post("/visualize/explain")
async def explain_visualization(req: VisualizeExplainWebRequest) -> dict:
    """Given a visualization code, provide an explanation of the code"""
    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(n=1, temperature=0)
    try:
        explanations = lida.explain(code=req.code, textgen_config=textgen_config, library=req.library)
        return {
            "status": True,
            "explanations": explanations[0],
            "message": "Successfully generated explanations",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization explanation: {str(exception_error)}")
        return {
            "status": False,
            "message": "Error generating visualization explanation.",
        }


@api.post("/visualize/evaluate")
async def evaluate_visualization(req: VisualizeEvalWebRequest) -> dict:
    """Given a visualization code, provide an evaluation of the code"""

    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(n=1, temperature=0)
        evaluations = lida.evaluate(
            code=req.code,
            goal=req.goal,
            textgen_config=textgen_config,
            library=req.library,
        )[0]
        return {
            "status": True,
            "evaluations": evaluations,
            "message": "Successfully generated evaluation",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization evaluation: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error generating visualization evaluation. {str(exception_error)}",
        }


@api.post("/visualize/recommend")
async def recommend_visualization(req: VisualizeRecommendRequest) -> dict:
    """Given a dataset summary, generate a visualization recommendations"""

    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.recommend(
            summary=req.summary,
            code=req.code,
            textgen_config=textgen_config,
            library=req.library,
            return_error=True,
        )

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {
            "status": True,
            "charts": charts,
            "message": "Successfully generated chart recommendation",
        }

    except Exception as exception_error:
        logger.error(f"Error generating visualization recommendation: {str(exception_error)}")
        return {
            "status": False,
            "message": "Error generating visualization recommendation.",
        }


@api.post("/text/generate")
async def generate_text(textgen_config: TextGenerationConfig) -> dict:
    """Generate text given some prompt"""

    try:
        completions = textgen.generate(textgen_config)
        return {"status": True, "completions": completions.text}
    except Exception as exception_error:
        logger.error(f"Error generating text: {str(exception_error)}")
        return {"status": False, "message": "Error generating text."}


@api.post("/goal")
async def generate_goal(req: GoalWebRequest) -> dict:
    """Generate goals given a dataset summary"""
    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        goals = lida.goals(req.summary, n=req.n, textgen_config=textgen_config, method=req.method)
        return {
            "status": True,
            "data": goals,
            "message": f"Successfully generated {len(goals)} goals",
        }
    except Exception as exception_error:
        logger.error(f"Error generating goals: {str(exception_error)}")
        # Check for a specific error message related to context length
        if "context length" in str(exception_error).lower():
            return {
                "status": False,
                "message": "The dataset you uploaded has too many columns. Please upload a dataset with fewer columns and try again.",
            }

        # For other exceptions
        return {
            "status": False,
            "message": f"Error generating visualization goals. {exception_error}",
        }


@api.post("/goal/stream")
async def generate_goal_stream(req: GoalWebRequest):
    """Streaming variant of /goal that pushes stage + llm.token events.

    Body matches /goal exactly. The response is text/event-stream and emits
    the same shape as /summarize/stream: ``stage``, ``llm.token``, ``goals``,
    ``complete``, ``done``. The frontend can consume both endpoints with the
    same SSE parser.
    """
    def run(emit: EmitFn) -> None:
        _run_goal_stream(req, emit)

    return StreamingResponse(
        _sse_streaming_response(run),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class _GoalStreamParser:
    """Incremental JSON-array parser that yields goal objects as they close.

    Walks the streamed text once, tracking brace depth, string state, and
    the position of the open ``{`` for each top-level object. When depth
    returns to zero, the closed substring is parsed as JSON and yielded.
    State is preserved across feed() calls so we don't rescan from the
    beginning each time.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.scan = 0
        self.array_open = False
        self.depth = 0
        self.in_str = False
        self.escape = False
        self.obj_start = -1

    def feed(self, delta: str) -> List[Dict]:
        self.buffer += delta
        out: List[Dict] = []
        n = len(self.buffer)
        i = self.scan
        while i < n:
            c = self.buffer[i]
            if not self.array_open:
                if c == "[":
                    self.array_open = True
                i += 1
                continue
            if self.in_str:
                if self.escape:
                    self.escape = False
                elif c == "\\":
                    self.escape = True
                elif c == '"':
                    self.in_str = False
            else:
                if c == '"':
                    self.in_str = True
                elif c == "{":
                    if self.depth == 0:
                        self.obj_start = i
                    self.depth += 1
                elif c == "}":
                    self.depth -= 1
                    if self.depth == 0 and self.obj_start >= 0:
                        snippet = self.buffer[self.obj_start : i + 1]
                        try:
                            out.append(json.loads(snippet))
                        except json.JSONDecodeError:
                            # Partial JSON; the final adapter parse will catch
                            # whatever the streaming pass missed.
                            pass
                        self.obj_start = -1
                elif c == "]" and self.depth == 0:
                    # End of array; stop scanning.
                    i = n
                    break
            i += 1
        self.scan = i
        return out


def _run_goal_stream(req: GoalWebRequest, emit: EmitFn) -> None:
    """Synchronous goal-generation pipeline body for the SSE endpoint.

    Parses streamed tokens incrementally so each goal is surfaced via a
    ``goal.ready`` event the moment its closing brace arrives — the user
    sees goals appear one at a time instead of waiting for the full array.
    """
    emit("stage", {"stage": "goals", "status": "start", "label": "Generating goals"})

    token_state = {"count": 0, "started_at": time.time()}
    parser = _GoalStreamParser()
    streamed_count = {"n": 0}

    def on_token(delta: str) -> None:
        token_state["count"] += 1
        emit(
            "llm.token",
            {
                "delta": delta,
                "tokens": token_state["count"],
                "elapsed_ms": int((time.time() - token_state["started_at"]) * 1000),
            },
        )
        for goal in parser.feed(delta):
            emit(
                "goal.ready",
                {
                    "index": streamed_count["n"],
                    "goal": jsonable_encoder(goal),
                },
            )
            streamed_count["n"] += 1

    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
    method = req.method or "default"

    try:
        with token_sink(on_token):
            goals = lida.goals(
                req.summary,
                n=req.n,
                textgen_config=textgen_config,
                method=method,
            )
    except Exception as exc:
        logger.error("goal stream failed: %s", exc, exc_info=True)
        msg = str(exc)
        if "context length" in msg.lower():
            msg = "The dataset has too many columns. Please reduce columns and try again."
        emit("stage", {"stage": "goals", "status": "error", "message": msg})
        emit("error", {"message": msg})
        return

    logger.info(
        "goal stream: %d goals generated (%d tokens, %d streamed incrementally)",
        len(goals), token_state["count"], streamed_count["n"],
    )
    # Final authoritative list — emitted after the adapter's own JSON parse,
    # which has more lenient cleanup than our streaming brace tracker.
    emit("goals", {"data": jsonable_encoder(goals)})
    emit("stage", {"stage": "goals", "status": "done", "tokens_total": token_state["count"]})
    emit(
        "complete",
        {
            "data": {
                "status": True,
                "data": jsonable_encoder(goals),
                "message": f"Successfully generated {len(goals)} goals",
            }
        },
    )


@api.post("/summarize")
async def upload_file(
    file: UploadFile,
    visualize: bool = os.environ.get("LIDA_MO_INSTANT_ANALYSIS", "true").lower() == "true",
):
    """Upload a file and return a summary of the data.

    Tabular files (csv/json/excel/parquet/feather) are summarized as a single
    dataset. SQLite uploads are unpacked into one summary per user table; the
    response includes a ``tables`` list with per-table summaries (and goals/charts
    when ``visualize`` is true).
    """

    try:
        saved_path = os.path.join(data_folder, file.filename or "upload.bin")
        with open(saved_path, "wb+") as file_object:
            file_object.write(file.file.read())

        file_location, display_name = _decompress_if_gzipped(saved_path, file.filename or "upload.bin")

        textgen_config = TextGenerationConfig(n=1, temperature=0)
        entries = _process_uploaded_file(file_location, display_name, textgen_config, visualize)

        if not entries:
            return {"status": False, "message": "No tabular data found in upload."}
        if len(entries) == 1 and "table_name" not in entries[0]:
            return {"status": True, **entries[0]}
        return {
            "status": True,
            "is_database": True,
            "data_filename": display_name,
            "tables": entries,
        }
    except Exception as exception_error:
        logger.error(f"Error processing file: {str(exception_error)}")
        logger.debug(traceback.format_exc())
        return {"status": False, "message": "Error processing file."}


@api.post("/summarize/stream")
async def upload_file_stream(
    file: UploadFile,
    visualize: bool = os.environ.get("LIDA_MO_INSTANT_ANALYSIS", "true").lower() == "true",
    n_goals: int = 5,
):
    """Streaming variant of /summarize that pushes per-stage progress events.

    Response body is ``text/event-stream``. The frontend reads the response
    incrementally (XHR readyState=3 or fetch ReadableStream) and renders the
    pipeline tracker. Events come in two top-level shapes:

    - ``stage`` — one of compress/upload/decompress/analyze/visualize. Always
      paired (``status: "start"`` then ``status: "done"`` or ``"error"``).
    - ``payload`` — domain events: ``summary.ready``, ``goals.ready``,
      ``chart.rendered``, ``llm.token``, etc. The final event is
      ``{"event": "complete", ...}`` followed by ``{"event": "done"}``.
    """

    upload_filename = file.filename or "upload.bin"
    saved_path = os.path.join(data_folder, upload_filename)
    # Read the body fully before we start streaming the response. FastAPI has
    # already buffered it; doing this synchronously keeps the upload phase
    # discrete from the analysis phase the client will see in the tracker.
    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    n_goals = max(1, min(10, int(n_goals)))

    return StreamingResponse(
        _stream_pipeline(saved_path, upload_filename, visualize, n_goals),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when proxied
        },
    )


def _sse_frame(event: str, payload: dict) -> bytes:
    body = {"event": event, "ts": time.time(), **payload}
    return f"data: {json.dumps(body, default=str)}\n\n".encode("utf-8")


SSE_KEEPALIVE_INTERVAL_S = 5.0


async def _sse_streaming_response(run_sync: Callable[[EmitFn], None]):
    """Generic SSE wrapper.

    ``run_sync`` is a synchronous callable that takes an ``emit(event, payload)``
    function and produces work. We run it in a worker thread, drain its emitted
    events through an asyncio.Queue, and yield SSE frames. A ``: keepalive``
    comment every few seconds prevents buffering proxies (nginx, Vite, k8s
    ingress) from holding bytes back during quiet stretches.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    def emit(event: str, payload: dict) -> None:
        # Worker thread -> async queue. call_soon_threadsafe is required
        # because the sync pipeline runs in asyncio.to_thread(), not on the
        # event loop.
        loop.call_soon_threadsafe(queue.put_nowait, (event, payload))

    def run() -> None:
        try:
            run_sync(emit)
        except Exception as exc:
            logger.error("SSE pipeline crashed: %s", exc, exc_info=True)
            emit("error", {"message": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    pipeline_task = asyncio.create_task(asyncio.to_thread(run))

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL_S)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            if item is sentinel:
                yield _sse_frame("done", {})
                return
            event, payload = item
            yield _sse_frame(event, payload)
    finally:
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass


async def _stream_pipeline(saved_path: str, upload_filename: str, visualize: bool, n_goals: int):
    def run(emit: EmitFn) -> None:
        _run_streaming_pipeline(saved_path, upload_filename, visualize, n_goals, emit)
    async for chunk in _sse_streaming_response(run):
        yield chunk


def _run_streaming_pipeline(
    saved_path: str,
    upload_filename: str,
    visualize: bool,
    n_goals: int,
    emit: EmitFn,
) -> None:
    """Synchronous pipeline body, called inside asyncio.to_thread().

    Emits stage and payload events through ``emit``. token_sink wires LLM token
    deltas into the same stream as ``llm.token`` events for the active stage.
    """
    emit("stage", {"stage": "decompress", "status": "start", "label": "Decompressing"})
    file_location, display_name = _decompress_if_gzipped(saved_path, upload_filename)
    emit(
        "stage",
        {
            "stage": "decompress",
            "status": "done",
            "display_name": display_name,
        },
    )

    textgen_config = TextGenerationConfig(n=1, temperature=0)

    emit("stage", {"stage": "analyze", "status": "start", "label": "Analyzing dataset"})

    # Token sinks are now installed per-table inside summarize_and_visualize so
    # parallel summarizers tag tokens with file_name. We just run the dispatch.
    try:
        entries = _process_uploaded_file(
            file_location, display_name, textgen_config, visualize, emit=emit, n_goals=n_goals,
        )
    except Exception as exc:
        emit("stage", {"stage": "analyze", "status": "error", "message": str(exc)})
        raise

    emit("stage", {"stage": "analyze", "status": "done"})
    logger.info("pipeline: analyze stage complete, building final payload")

    if not entries:
        emit("error", {"message": "No tabular data found in upload."})
        return

    if len(entries) == 1 and "table_name" not in entries[0]:
        complete_payload = {"status": True, **entries[0]}
    else:
        complete_payload = {
            "status": True,
            "is_database": True,
            "data_filename": display_name,
            "tables": entries,
        }

    emit("complete", {"data": jsonable_encoder(complete_payload)})
    logger.info("pipeline: emitted complete event")


def _decompress_if_gzipped(saved_path: str, original_filename: str) -> tuple[str, str]:
    """If the uploaded file is gzip-wrapped (detected by magic bytes), unwrap it.

    Returns (canonical_path, canonical_filename). The wrapped file is removed so
    downstream callers see the unwrapped file. ``.tgz`` is mapped to ``.tar``.
    """
    if not is_gzip_file(saved_path):
        return saved_path, original_filename

    lower = original_filename.lower()
    if lower.endswith(".tgz"):
        inner_name = original_filename[:-4] + ".tar"
    elif lower.endswith(".gz"):
        inner_name = original_filename[:-3]
    else:
        inner_name = original_filename + ".out"
    if not inner_name:
        return saved_path, original_filename

    inner_path = os.path.join(data_folder, inner_name)
    with gzip.open(saved_path, "rb") as src, open(inner_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    try:
        os.unlink(saved_path)
    except OSError:
        pass
    return inner_path, inner_name


def _process_uploaded_file(
    file_location: str,
    display_name: str,
    textgen_config: TextGenerationConfig,
    run_visualize: bool,
    emit: Optional[EmitFn] = None,
    n_goals: int = 5,
) -> list:
    """Dispatch on file kind, returning a flat list of per-dataset result dicts."""
    if is_sqlite_file(file_location):
        _emit(emit, "dispatch", kind="sqlite", file_name=display_name)
        return _sqlite_entries(file_location, display_name, textgen_config, run_visualize, emit=emit, n_goals=n_goals)
    if is_tar_archive(file_location):
        _emit(emit, "dispatch", kind="tar", file_name=display_name)
        return _tar_entries(file_location, display_name, textgen_config, run_visualize, emit=emit, n_goals=n_goals)
    _emit(emit, "dispatch", kind="single", file_name=display_name)
    return [summarize_and_visualize(file_location, display_name, textgen_config, run_visualize, emit=emit, n_goals=n_goals)]


TABLE_CONCURRENCY = max(1, int(os.environ.get("LIDA_TABLE_CONCURRENCY", "4")))


def _run_tables_in_parallel(
    units: list,
    textgen_config: TextGenerationConfig,
    run_visualize: bool,
    emit: Optional[EmitFn],
    n_goals: int = 5,
) -> list:
    """Fan out summarize_and_visualize across multiple tables concurrently.

    ``units`` is a list of tuples ``(label, csv_path, display_filename)`` where
    ``label`` is the per-table name surfaced through table.started/done events.
    Concurrency is bounded by LIDA_TABLE_CONCURRENCY (default 4); per-table
    chart pools are sized to share CPU rather than oversubscribe.
    """
    n = len(units)
    if n == 0:
        return []
    parallelism = min(n, TABLE_CONCURRENCY)
    chart_workers_cap = max(1, (os.cpu_count() or 4) // parallelism)
    logger.info(
        "pipeline: dispatching %d tables across %d workers (chart cap %d/table)",
        n, parallelism, chart_workers_cap,
    )

    results: dict = {}

    def worker(label: str, csv_path: str, display_filename: str) -> tuple:
        _emit(emit, "table.started", name=label)
        try:
            entry = summarize_and_visualize(
                csv_path,
                display_filename,
                textgen_config,
                run_visualize,
                emit=emit,
                chart_workers_cap=chart_workers_cap,
                n_goals=n_goals,
            )
            entry["table_name"] = label
            _emit(emit, "table.done", name=label)
            return label, entry, None
        except Exception as exc:
            logger.warning("table %s failed: %s", label, exc)
            _emit(emit, "table.error", name=label, error=str(exc))
            return label, None, exc

    with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="lida-table") as ex:
        futures = [ex.submit(worker, *u) for u in units]
        for fut in as_completed(futures):
            label, entry, exc = fut.result()
            if entry is not None:
                results[label] = entry

    # Preserve original input order in the response.
    ordered = []
    for label, _, _ in units:
        if label in results:
            ordered.append(results[label])
    return ordered


def _sqlite_entries(
    file_location: str,
    db_filename: str,
    textgen_config: TextGenerationConfig,
    run_visualize: bool,
    emit: Optional[EmitFn] = None,
    n_goals: int = 5,
) -> list:
    """Materialize each user table as a CSV, then run the standard pipeline per table."""
    tables = read_sqlite_tables(file_location)
    if not tables:
        return []
    _emit(
        emit,
        "tables.detected",
        kind="sqlite",
        count=len(tables),
        names=list(tables.keys()),
    )
    stem, _ = os.path.splitext(db_filename)
    units = []
    for table_name, df in tables.items():
        csv_filename = f"{stem}__{table_name}.csv"
        csv_path = os.path.join(data_folder, csv_filename)
        df.to_csv(csv_path, index=False)
        units.append((table_name, csv_path, csv_filename))
    return _run_tables_in_parallel(units, textgen_config, run_visualize, emit, n_goals=n_goals)


def _tar_entries(
    file_location: str,
    archive_filename: str,
    textgen_config: TextGenerationConfig,
    run_visualize: bool,
    emit: Optional[EmitFn] = None,
    n_goals: int = 5,
) -> list:
    """Extract each regular file from a tar archive and run the pipeline.

    Sanitizes member names to defeat path traversal and skips directories,
    symlinks, and dotfiles. Tabular members are run concurrently. Recursive
    members (a sqlite or another tar inside the archive) fall back to the
    sequential dispatch and are appended in extraction order.
    """
    units: list = []
    sequential_targets: list = []
    with tarfile.open(file_location, "r") as tf:
        members = [m for m in tf.getmembers() if m.isreg()]
        keep = [m for m in members if os.path.basename(m.name) and not os.path.basename(m.name).startswith(".")]
        names = [os.path.basename(m.name) for m in keep]
        _emit(emit, "tables.detected", kind="tar", count=len(names), names=names)
        for member in keep:
            base_name = os.path.basename(member.name)
            dest_path = os.path.join(data_folder, base_name)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(src, out)
            # Only tabular files run through summarize_and_visualize directly;
            # nested sqlite/tar archives need recursive dispatch and stay
            # sequential to keep the event ordering predictable.
            if is_sqlite_file(dest_path) or is_tar_archive(dest_path):
                sequential_targets.append((base_name, dest_path))
            else:
                units.append((base_name, dest_path, base_name))

    parallel_results = _run_tables_in_parallel(units, textgen_config, run_visualize, emit, n_goals=n_goals)

    sequential_results: list = []
    for base_name, dest_path in sequential_targets:
        _emit(emit, "table.started", name=base_name)
        try:
            sub_entries = _process_uploaded_file(
                dest_path, base_name, textgen_config, run_visualize, emit=emit, n_goals=n_goals,
            )
        except Exception as exc:
            logger.warning(f"Skipping tar entry {base_name}: {exc}")
            _emit(emit, "table.error", name=base_name, error=str(exc))
            continue
        for entry in sub_entries:
            entry.setdefault("table_name", base_name)
            sequential_results.append(entry)
        _emit(emit, "table.done", name=base_name)

    return parallel_results + sequential_results


# upload via url
@api.post("/summarize/url")
async def upload_file_via_url(req: SummaryUrlRequest) -> dict:
    """Upload a file from a url and return a summary of the data"""
    url = req.url
    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(n=1, temperature=0)
    file_name = url.split("/")[-1]
    file_location = os.path.join(data_folder, file_name)

    # download file
    url_response = requests.get(url, allow_redirects=True, timeout=1000)
    open(file_location, "wb").write(url_response.content)
    try:
        summary = lida.summarize(
            data=file_location,
            file_name=file_name,
            summary_method="llm",
            textgen_config=textgen_config,
        )
        response = {"status": True, "summary": summary, "data_filename": file_name}

        if req.visualize:
            # Automatic visualization flow (Heuristic)
            goals = lida.goals(summary, n=20, textgen_config=textgen_config, method="heuristic")
            # Generate charts for these goals
            charts = []
            for i, goal in enumerate(goals):
                # Generate code
                code_list = lida.visualize(
                    summary=summary,
                    goal=goal,
                    textgen_config=textgen_config,
                    library="seaborn",
                    method="heuristic",
                    return_error=True,
                )
                if code_list:
                    code = code_list[0]  # Heuristic returns list of 1
                    # Execute code
                    # Read dataframe for execution
                    if i == 0:
                        data_df = read_dataframe(file_location)

                    chart_executed = lida.execute(
                        code_specs=[code],
                        data=data_df,
                        summary=summary,
                        library="seaborn",
                    )
                    if chart_executed and len(chart_executed) > 0:
                        charts.append(chart_executed[0])

            response["goals"] = goals
            response["charts"] = charts

        return response
    except Exception as exception_error:
        logger.error(f"Error processing file from URL: {str(exception_error)}")
        logger.debug(traceback.format_exc())
        return {"status": False, "message": "Error processing file."}


# convert image to infographics


@api.post("/infographer")
async def generate_infographics(req: InfographicsRequest) -> dict:
    """Generate infographics using the peacasso package"""
    try:
        result = lida.infographics(
            visualization=req.visualization,
            n=req.n,
            style_prompt=req.style_prompt,
            # return_pil=req.return_pil
        )
        return {
            "status": True,
            "result": result,
            "message": "Successfully generated infographics",
        }
    except Exception as exception_error:
        logger.error(f"Error generating infographics: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error generating infographics. {str(exception_error)}",
        }


@api.get("/models")
def list_models() -> dict:
    return {
        "status": True,
        "data": {
            "litellm": {
                "name": "litellm",
                "description": "LiteLLM proxy (OpenAI-compatible)",
                "base_url": textgen.base_url,
                "models": [{"name": textgen.model_name}],
            }
        },
        "default": "litellm",
        "message": "Successfully listed models",
    }
