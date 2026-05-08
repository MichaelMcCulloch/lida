import os
import logging
import requests
import traceback
import multiprocessing
import matplotlib

matplotlib.use("Agg")  # Ensure headless execution
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from lida.utils import read_dataframe

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
    """Upload a file and return a summary of the data"""

    # allow csv, excel, json

    # if file.content_type not in allowed_types:
    #     return {
    #         "status": False,
    #         "message": f"Uploaded file type ({file.content_type}) not allowed. Allowed types are: csv, xls, xlsx, json",
    #     }

    try:
        # save file to files folder
        file_location = os.path.join(data_folder, file.filename)
        # open file without deleting existing contents
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # summarize
        textgen_config = TextGenerationConfig(n=1, temperature=0)
        summary = lida.summarize(
            data=file_location,
            file_name=file.filename,
            summary_method="llm",
            textgen_config=textgen_config,
        )

        response = {"status": True, "summary": summary, "data_filename": file.filename}

        print(f"DEBUG: Summary generated. Visualize flag: {visualize}")

        if visualize:
            # Automatic visualization flow (Heuristic)
            print("DEBUG: Automatic heuristic visualization triggered")
            try:
                goals = lida.goals(summary, n=20, textgen_config=textgen_config, method="heuristic")
                print(f"DEBUG: Generated {len(goals)} heuristic goals")
                # Generate charts for these goals
                charts = []

                # Step 1: Generate code for all goals first (sequential, low latency)
                goals_with_code = []
                print("DEBUG: Generating code for all goals...")
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
                        print(f"DEBUG: Failed to generate code for goal {goal.index}")

                # Step 2: Execute charts in parallel
                print(f"DEBUG: Executing {len(goals_with_code)} charts in parallel...")
                if len(goals_with_code) > 0:
                    data_df = read_dataframe(file_location)  # Read once

                    # Use ProcessPoolExecutor for parallel execution
                    # Adjust max_workers as needed, default to None (os.cpu_count()) or a safe number
                    max_workers = min(len(goals_with_code), os.cpu_count() or 4)

                    # Use 'spawn' context to avoid gRPC (Gemini) deadlocks in forked processes
                    ctx = multiprocessing.get_context("spawn")
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                        # Map futures to their index to maintain order if needed, or just collect
                        future_to_index = {executor.submit(execute_chart_worker, code, data_df, summary, "seaborn"): (i, goal) for i, goal, code in goals_with_code}

                        # Collect results as they complete
                        metrics_executed = 0

                        # We need to reconstruct the order or just append.
                        # The UI expects 'charts' list. If order matters (matching goals?),
                        # we should probably effectively re-sort or returning a map.
                        # Goals are returned in 'goals' list. Charts are in 'charts' list.
                        # The frontend likely maps charts[i] to goals[i] if indices match,
                        # OR the goal object is embedded.
                        # In the original code:
                        #   for i, goal in enumerate(goals): ... charts.append(chart_executed[0])
                        # failed execution -> no chart appended?
                        # actually, if chart_executed is empty, nothing appended.
                        # This implies 'charts' length might be < 'goals' length.
                        # Let's verify existing logic:
                        # if chart_executed: charts.append(chart_executed[0])
                        # else: print (failed)
                        # So charts list corresponds to SUCCESSFUL charts, but losing alignment with 'goals' list.
                        # If the frontend relies on index alignment, the original code is buggy if any chart fails.
                        # Let's assume we just want valid charts.

                        # However, for consistency with parallel execution, we should try to keep order or just append valid ones.

                        results = []
                        for future in as_completed(future_to_index):
                            i, goal = future_to_index[future]
                            try:
                                chart_executed = future.result()
                                if chart_executed and len(chart_executed) > 0:
                                    results.append((i, chart_executed[0]))
                                    metrics_executed += 1
                                else:
                                    print(f"DEBUG: Worker returned no chart for goal {goal.index}")
                            except Exception as exc:
                                print(f"DEBUG: Chart execution generated an exception for goal {goal.index}: {exc}")

                        # Sort by original index to maintain heuristic order (importance)
                        results.sort(key=lambda x: x[0])
                        charts = [r[1] for r in results]

                        print(f"DEBUG: Successfully executed {metrics_executed} charts")

                print(f"DEBUG: Total charts generated: {len(charts)}")
                response["goals"] = goals
                response["charts"] = charts
            except Exception as e:
                print(f"DEBUG: Error in heuristic flow: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("DEBUG: Visualization skipped (visualize=False)")

        return response
    except Exception as exception_error:
        logger.error(f"Error processing file: {str(exception_error)}")
        return {"status": False, "message": "Error processing file."}


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
