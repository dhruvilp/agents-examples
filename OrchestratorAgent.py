import asyncio
import logging
import json
from typing import AsyncGenerator, List, Dict, Any

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from typing_extensions import override

# --- Configuration and Constants ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-flash"
APP_NAME = "OrchestratorApp"
USER_ID = "test_user_456"

# --- State Keys for Session Management ---
# Using constants for state keys prevents typos and makes the code more maintainable.
STATE_OBJECTIVE = "objective"
STATE_PLAN = "plan"
STATE_EXECUTED_STEPS = "executed_steps"
STATE_LAST_EXECUTION_RESULT = "last_execution_result"
STATE_DECISION = "decision"
STATE_IS_DONE = "is_done"
STATE_CURRENT_STEP = "current_step"


# --- Tool Definition ---
def web_search(query: str) -> Dict[str, Any]:
    """
    Performs a web search for the given query.

    Args:
        query: The search query.

    Returns:
        A dictionary containing the search result.
    """
    logger.info(f"TOOL CALL: web_search(query='{query}')")
    # In a real application, this would call a search API (e.g., Google Search API).
    # For this example, we return a mock result for demonstration purposes.
    if "capital of France" in query.lower():
        return {"result": "The capital of France is Paris."}
    if "python programming language" in query.lower():
        return {
            "result": "Python is a high-level, general-purpose programming language known for its readability."
        }
    return {
        "result": f"No specific result found for '{query}'. The search tool is a mock."
    }


# --- Sub-Agent Class Definitions ---


class PlannerAgent(LlmAgent):
    """
    An agent responsible for creating a step-by-step plan to achieve an objective.
    It can also be invoked for replanning if the initial plan fails.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="PlannerAgent",
            model=model,
            instruction=f"""
            You are an expert planner. Your role is to create a clear, concise, and
            step-by-step plan to achieve a given objective.

            **Input:**
            - An 'objective'.
            - Optionally, 'executed_steps' and 'last_execution_result' for context if replanning.

            **Task:**
            1. Analyze the objective.
            2. If this is a replan, review the previous steps and the result of the last one to create a better plan.
            3. Create a JSON object containing a list of strings named "plan". Each string is a clear, actionable step.
            4. The plan should be logical and sequential. Do not create overly complex plans.

            **Example Output:**
            {{
              "plan": [
                "Use the web_search tool to find the capital of France.",
                "State the final answer based on the search result."
              ]
            }}

            **Current State:**
            - Objective: {{{STATE_OBJECTIVE}}}
            - Executed Steps: {{{STATE_EXECUTED_STEPS}}}
            - Last Execution Result: {{{STATE_LAST_EXECUTION_RESULT}}}
            """,
            output_key=STATE_PLAN,
            response_mime_type="application/json",
        )


class ExecutionAgent(LlmAgent):
    """
    An agent responsible for executing a single step from a plan,
    using available tools if necessary.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="ExecutionAgent",
            model=model,
            tools=[FunctionTool.from_function(web_search)],
            instruction=f"""
            You are an execution agent. Your role is to execute a single, specific step from a plan.

            **Input:**
            - The full 'plan'.
            - The specific 'current_step' to execute.

            **Task:**
            1. Read the 'current_step'.
            2. If a tool is required by the step (e.g., 'web_search'), call it with the correct parameters.
            3. Output the result of the execution. This result will be reviewed by the orchestrator.
               Your output should be a direct and factual summary of what you did or found.

            **Current State:**
            - Plan: {{{STATE_PLAN}}}
            - Step to Execute: {{{STATE_CURRENT_STEP}}}
            """,
            output_key=STATE_LAST_EXECUTION_RESULT,
        )


class CoordinatorAgent(LlmAgent):
    """
    An agent that analyzes the current state of the task and decides the next action.
    It acts as the "brain" of the orchestrator's loop.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="CoordinatorAgent",
            model=model,
            instruction=f"""
            You are the coordinator. Your role is to analyze the progress and decide the next action.

            **Input:**
            - The 'objective'.
            - The 'plan'.
            - The 'executed_steps'.
            - The 'last_execution_result'.

            **Task:**
            Analyze the inputs and determine the state of the task. Respond with a JSON object
            containing your decision.

            **Decision Logic:**
            1.  If 'last_execution_result' clearly and completely fulfills the 'objective',
                set 'next_action' to 'FINISH' and 'is_done' to true.
            2.  If the plan is not yet fully executed, find the next step that is not in 'executed_steps'.
                Set 'next_action' to 'EXECUTE' and 'next_step' to that step.
            3.  If all steps in the plan are executed but the objective is not met, or if the
                'last_execution_result' indicates a significant problem or error, set 'next_action' to 'REPLAN'.
            4.  If there is no plan yet ('plan' is null or empty), set 'next_action' to 'PLAN'.

            **Example Output:**
            {{
              "thought": "The first step was executed successfully. Now proceeding to the next step.",
              "next_action": "EXECUTE",
              "next_step": "State the final answer based on the search result.",
              "is_done": false
            }}

            **Current State:**
            - Objective: {{{STATE_OBJECTIVE}}}
            - Plan: {{{STATE_PLAN}}}
            - Executed Steps: {{{STATE_EXECUTED_STEPS}}}
            - Last Execution Result: {{{STATE_LAST_EXECUTION_RESULT}}}
            """,
            output_key=STATE_DECISION,
            response_mime_type="application/json",
        )


# --- Orchestrator Agent Definition ---


class OrchestratorAgent(BaseAgent):
    """
    A custom orchestrator agent that manages a workflow of planning and execution
    to achieve a user-defined objective by coordinating Planner, Executor, and Coordinator agents.
    """

    model_config = {"arbitrary_types_allowed": True}

    planner: PlannerAgent
    executor: ExecutionAgent
    coordinator: CoordinatorAgent

    def __init__(
        self, planner: PlannerAgent, executor: ExecutionAgent, coordinator: CoordinatorAgent
    ):
        """Initializes the OrchestratorAgent with its sub-agents."""
        super().__init__(
            name="OrchestratorAgent",
            sub_agents=[planner, executor, coordinator],
            planner=planner,
            executor=executor,
            coordinator=coordinator,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic.
        This method defines the main loop of: Decide -> Plan/Execute -> Repeat.
        """
        logger.info(f"[{self.name}] Starting orchestration workflow.")

        # Initialize state from the invocation context
        if ctx.new_message:
            ctx.session.state[STATE_OBJECTIVE] = ctx.new_message.stringify_content()
        ctx.session.state.setdefault(STATE_PLAN, None)
        ctx.session.state.setdefault(STATE_EXECUTED_STEPS, [])
        ctx.session.state.setdefault(STATE_LAST_EXECUTION_RESULT, None)
        ctx.session.state.setdefault(STATE_IS_DONE, False)

        max_turns = 10
        for turn_count in range(max_turns):
            logger.info(f"--- Turn {turn_count + 1}/{max_turns} ---")

            # 1. DECIDE the next action by running the CoordinatorAgent
            logger.info(f"[{self.name}] Running CoordinatorAgent...")
            async for event in self.coordinator.run_async(ctx):
                yield event

            decision_str = ctx.session.state.get(STATE_DECISION, "{}")
            try:
                decision = json.loads(decision_str)
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Failed to parse decision JSON: {decision_str}")
                yield Event.from_error(
                    author=self.name,
                    error_message="Coordinator failed to produce valid JSON. Aborting.",
                )
                break

            logger.info(f"[{self.name}] Decision: {decision.get('thought')}")

            if decision.get("is_done"):
                logger.info(f"[{self.name}] Objective achieved. Workflow finished.")
                final_result = ctx.session.state.get(
                    STATE_LAST_EXECUTION_RESULT, "Task completed successfully."
                )
                yield Event.from_final_response(
                    author=self.name, content=str(final_result)
                )
                break

            next_action = decision.get("next_action")

            # 2. ACT based on the decision
            if next_action in ("PLAN", "REPLAN"):
                logger.info(f"[{self.name}] Running PlannerAgent to {next_action.lower()}...")
                if next_action == "REPLAN":
                    # Reset execution context for the new plan
                    ctx.session.state[STATE_EXECUTED_STEPS] = []
                    ctx.session.state[STATE_LAST_EXECUTION_RESULT] = None
                async for event in self.planner.run_async(ctx):
                    yield event

            elif next_action == "EXECUTE":
                step_to_execute = decision.get("next_step")
                if not step_to_execute:
                    logger.error("Coordinator chose EXECUTE but provided no step. Aborting.")
                    yield Event.from_error(
                        author=self.name,
                        error_message="Coordinator failed to provide next step. Aborting.",
                    )
                    break

                logger.info(f"[{self.name}] Running ExecutionAgent for step: '{step_to_execute}'")
                ctx.session.state[STATE_CURRENT_STEP] = step_to_execute
                async for event in self.executor.run_async(ctx):
                    yield event

                # Update list of executed steps
                ctx.session.state[STATE_EXECUTED_STEPS].append(step_to_execute)

            else:
                logger.error(f"[{self.name}] Unknown next action: '{next_action}'. Aborting.")
                yield Event.from_error(
                    author=self.name,
                    error_message=f"Unknown action '{next_action}'. Aborting.",
                )
                break
        else:
            logger.warning(f"[{self.name}] Max turns ({max_turns}) reached. Aborting.")
            yield Event.from_error(
                author=self.name,
                error_message=f"Max turns ({max_turns}) reached without completion.",
            )


async def main():
    """
    Sets up the runner and session, then runs the OrchestratorAgent with a sample objective.
    """
    # --- Instantiate Agents from their classes ---
    planner = PlannerAgent()
    executor = ExecutionAgent()
    coordinator = CoordinatorAgent()
    orchestrator = OrchestratorAgent(planner, executor, coordinator)

    # --- Setup Runner and Session ---
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Run the Agent with an Objective ---
    objective = "What is the capital of France?"
    logger.info(f"--- Starting New Run ---")
    logger.info(f"Objective: {objective}")

    initial_message = genai_types.Content(
        role="user", parts=[genai_types.Part(text=objective)]
    )

    session_id = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    # --- Stream and Log Events from the Run ---
    final_response = "No final response was captured."
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=initial_message
    ):
        if event.is_final_response() and event.content:
            final_response = event.content.stringify_content()
            logger.info(f"====== FINAL RESPONSE from [{event.author}] ======\n{final_response}")
        elif event.is_error():
            logger.error(f"====== ERROR from [{event.author}] ======\n{event.error_message}")


    # --- Inspect Final State for debugging ---
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    logger.info("\n--- Final Session State ---")
    # Pretty print the final state JSON
    print(json.dumps(json.loads(final_session.model_dump_json()), indent=2))
    logger.info("--- Run Finished ---\n")


if __name__ == "__main__":
    # To run this, ensure you have `google-adk` and `google-generativeai` installed
    # and your environment is authenticated with `gcloud auth application-default login`.
    asyncio.run(main())
