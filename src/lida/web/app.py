import gzip
import os
import logging
import requests
import shutil
import tarfile
import traceback
import multiprocessing
import matplotlib

matplotlib.use("Agg")  # Ensure headless execution
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from lida.utils import is_gzip_file, is_sqlite_file, is_tar_archive, read_dataframe, read_sqlite_tables

from ..components.litellm_generator import LiteLLMTextGenerator
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


def summarize_and_visualize(file_location: str, display_name: str, textgen_config: TextGenerationConfig, run_visualize: bool) -> dict:
    """Run the full summarize -> heuristic goals -> parallel chart pipeline for one tabular file.

    Returns a dict with summary, data_filename, and (if run_visualize) goals + charts.
    """
    summary = lida.summarize(
        data=file_location,
        file_name=display_name,
        summary_method="llm",
        textgen_config=textgen_config,
    )
    result = {"summary": summary, "data_filename": display_name}

    if not run_visualize:
        return result

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
            else:
                logger.info(f"Failed to generate code for goal {goal.index} (table={display_name})")

        charts: list = []
        if goals_with_code:
            data_df = read_dataframe(file_location)
            max_workers = min(len(goals_with_code), os.cpu_count() or 4)
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                future_to_index = {executor.submit(execute_chart_worker, code, data_df, summary, "seaborn"): (i, goal) for i, goal, code in goals_with_code}
                rendered = []
                for future in as_completed(future_to_index):
                    i, goal = future_to_index[future]
                    try:
                        chart_executed = future.result()
                        if chart_executed:
                            rendered.append((i, chart_executed[0]))
                    except Exception as exc:
                        logger.warning(f"Chart exception for goal {goal.index} (table={display_name}): {exc}")
                rendered.sort(key=lambda x: x[0])
                charts = [r[1] for r in rendered]
        result["goals"] = goals
        result["charts"] = charts
    except Exception as exc:
        logger.error(f"Heuristic flow error for {display_name}: {exc}")
        logger.debug(traceback.format_exc())
    return result


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


def _process_uploaded_file(file_location: str, display_name: str, textgen_config: TextGenerationConfig, run_visualize: bool) -> list:
    """Dispatch on file kind, returning a flat list of per-dataset result dicts."""
    if is_sqlite_file(file_location):
        return _sqlite_entries(file_location, display_name, textgen_config, run_visualize)
    if is_tar_archive(file_location):
        return _tar_entries(file_location, display_name, textgen_config, run_visualize)
    return [summarize_and_visualize(file_location, display_name, textgen_config, run_visualize)]


def _sqlite_entries(file_location: str, db_filename: str, textgen_config: TextGenerationConfig, run_visualize: bool) -> list:
    """Materialize each user table as a CSV, then run the standard pipeline per table."""
    tables = read_sqlite_tables(file_location)
    if not tables:
        return []
    stem, _ = os.path.splitext(db_filename)
    results: list = []
    for table_name, df in tables.items():
        csv_filename = f"{stem}__{table_name}.csv"
        csv_path = os.path.join(data_folder, csv_filename)
        df.to_csv(csv_path, index=False)
        per_table = summarize_and_visualize(csv_path, csv_filename, textgen_config, run_visualize)
        per_table["table_name"] = table_name
        results.append(per_table)
    return results


def _tar_entries(file_location: str, archive_filename: str, textgen_config: TextGenerationConfig, run_visualize: bool) -> list:
    """Extract each regular file from a tar archive and recursively process it.

    Sanitizes member names to defeat path traversal and skips directories,
    symlinks, and dotfiles.
    """
    results: list = []
    with tarfile.open(file_location, "r") as tf:
        for member in tf.getmembers():
            if not member.isreg():
                continue
            base_name = os.path.basename(member.name)
            if not base_name or base_name.startswith("."):
                continue
            dest_path = os.path.join(data_folder, base_name)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(src, out)
            try:
                sub_entries = _process_uploaded_file(dest_path, base_name, textgen_config, run_visualize)
            except Exception as exc:
                logger.warning(f"Skipping tar entry {base_name}: {exc}")
                continue
            for entry in sub_entries:
                entry.setdefault("table_name", base_name)
                results.append(entry)
    return results


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
