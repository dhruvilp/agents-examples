import asyncio
import logging
import json
from typing import AsyncGenerator, List, Dict, Any

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from typing_extensions import override

# --- Configuration and Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-flash"
APP_NAME = "OrchestratorApp"
USER_ID = "test_user_123"

# --- State Keys for Session Management ---
# These keys are used to store and retrieve data from the session state,
# enabling communication and statefulness between agent runs.
STATE_OBJECTIVE = "objective"
STATE_PLAN = "plan"
STATE_EXECUTED_STEPS = "executed_steps"
STATE_LAST_EXECUTION_RESULT = "last_execution_result"
STATE_DECISION = "decision"
STATE_IS_DONE = "is_done"

# --- Tool Definitions ---
# A simple search tool for the ExecutionAgent to use.
# In a real-world scenario, this would be a robust API call.
def web_search(query: str) -> Dict[str, Any]:
    """
    Performs a web search for the given query.

    Args:
        query: The search query.

    Returns:
        A dictionary containing the search result.
    """
    logger.info(f"TOOL CALL: web_search(query='{query}')")
    # In a real application, this would call a search API.
    # For this example, we return a mock result.
    if "capital of France" in query.lower():
        return {"result": "The capital of France is Paris."}
    if "python programming language" in query.lower():
        return {"result": "Python is a high-level, general-purpose programming language."}
    return {"result": f"No specific result found for '{query}'. The search tool is a mock."}

# --- Sub-Agent Definitions ---

def build_planner_agent() -> LlmAgent:
    """Builds the PlannerAgent."""
    return LlmAgent(
        name="PlannerAgent",
        model=GEMINI_MODEL,
        instruction=f"""
        You are an expert planner. Your role is to create a clear, concise, and
        step-by-step plan to achieve a given objective.

        **Input:**
        - An 'objective'.
        - Optionally, the 'executed_steps' and the 'last_execution_result' for context if replanning.

        **Task:**
        1. Analyze the objective.
        2. If this is a replan, review the previous steps and the result of the last one.
        3. Create a JSON array of strings, where each string is a clear, actionable step.
        4. The plan should be logical and sequential. Do not create overly complex plans.

        **Example Output:**
        {{
          "plan": [
            "Use the web_search tool to find the capital of France.",
            "State the final answer."
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

def build_execution_agent() -> LlmAgent:
    """Builds the ExecutionAgent."""
    return LlmAgent(
        name="ExecutionAgent",
        model=GEMINI_MODEL,
        tools=[FunctionTool.from_function(web_search)],
        instruction=f"""
        You are an execution agent. Your role is to execute a single step from a plan.

        **Input:**
        - The full 'plan'.
        - The specific 'step_to_execute'.

        **Task:**
        1. Read the 'step_to_execute'.
        2. If a tool is required (e.g., 'web_search'), call it with the correct parameters.
        3. Output the result of the execution. This result will be reviewed by the orchestrator.

        **Current State:**
        - Plan: {{{STATE_PLAN}}}
        - Step to Execute: {{{STATE_OBJECTIVE}}}
        """,
        output_key=STATE_LAST_EXECUTION_RESULT,
    )

def build_decider_agent() -> LlmAgent:
    """Builds the DeciderAgent."""
    return LlmAgent(
        name="DeciderAgent",
        model=GEMINI_MODEL,
        instruction=f"""
        You are the decider. Your role is to analyze the progress and decide the next action.

        **Input:**
        - The 'objective'.
        - The 'plan'.
        - The 'executed_steps'.
        - The 'last_execution_result'.

        **Task:**
        Analyze the inputs and determine the state of the task. Respond with a JSON object
        containing your decision.

        **Decision Logic:**
        1.  If the 'last_execution_result' clearly and completely fulfills the 'objective',
            set 'next_action' to 'FINISH' and 'is_done' to true.
        2.  If the plan is not fully executed, find the next step that is not in 'executed_steps'.
            Set 'next_action' to 'EXECUTE' and 'next_step' to that step.
        3.  If all steps in the plan are executed but the objective is not met, or if the
            'last_execution_result' indicates a problem, set 'next_action' to 'REPLAN'.
        4.  If the plan is empty or null, set 'next_action' to 'PLAN'.

        **Example Output:**
        {{
          "thought": "The first step was executed successfully. Now proceeding to the next step.",
          "next_action": "EXECUTE",
          "next_step": "State the final answer.",
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

################################################################################

# --- Orchestrator Agent Definition ---

class OrchestratorAgent(BaseAgent):
    """
    A custom orchestrator agent that manages a workflow of planning and execution
    to achieve a user-defined objective.
    """
    model_config = {"arbitrary_types_allowed": True}

    planner: LlmAgent
    executor: LlmAgent
    decider: LlmAgent

    def __init__(self, planner: LlmAgent, executor: LlmAgent, decider: LlmAgent):
        """Initializes the OrchestratorAgent with its sub-agents."""
        super().__init__(
            name="OrchestratorAgent",
            sub_agents=[planner, executor, decider],
            planner=planner,
            executor=executor,
            decider=decider,
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the agent.
        This method defines the main loop of deciding, planning, and executing.
        """
        logger.info(f"[{self.name}] Starting orchestration workflow.")

        # Initialize state variables if they don't exist
        ctx.session.state.setdefault(STATE_PLAN, None)
        ctx.session.state.setdefault(STATE_EXECUTED_STEPS, [])
        ctx.session.state.setdefault(STATE_LAST_EXECUTION_RESULT, None)
        ctx.session.state.setdefault(STATE_IS_DONE, False)

        # The user's initial message is the objective.
        if ctx.new_message:
             ctx.session.state[STATE_OBJECTIVE] = ctx.new_message.stringify_content()


        max_turns = 10
        for turn_count in range(max_turns):
            logger.info(f"--- Turn {turn_count + 1} ---")

            # 1. DECIDE the next action
            logger.info(f"[{self.name}] Running DeciderAgent...")
            async for event in self.decider.run_async(ctx):
                yield event

            decision_str = ctx.session.state.get(STATE_DECISION, "{}")
            try:
                decision = json.loads(decision_str)
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Failed to parse decision JSON: {decision_str}")
                # Potentially escalate an error or try to recover
                break

            logger.info(f"[{self.name}] Decision: {decision.get('thought')}")

            if decision.get("is_done"):
                logger.info(f"[{self.name}] Objective achieved. Workflow finished.")
                ctx.session.state[STATE_IS_DONE] = True
                # Yield a final event with the result
                final_result = ctx.session.state.get(STATE_LAST_EXECUTION_RESULT, "Task completed.")
                yield Event.from_final_response(author=self.name, content=str(final_result))
                break

            next_action = decision.get("next_action")

            # 2. ACT based on the decision
            if next_action == "PLAN" or next_action == "REPLAN":
                logger.info(f"[{self.name}] Running PlannerAgent to {next_action.lower()}...")
                async for event in self.planner.run_async(ctx):
                    yield event
                # Reset execution context after replanning
                ctx.session.state[STATE_EXECUTED_STEPS] = []
                ctx.session.state[STATE_LAST_EXECUTION_RESULT] = None

            elif next_action == "EXECUTE":
                step_to_execute = decision.get("next_step")
                if not step_to_execute:
                    logger.error("Decider chose to EXECUTE but provided no step.")
                    break
                
                logger.info(f"[{self.name}] Running ExecutionAgent for step: '{step_to_execute}'")
                # We use the objective key to pass the step to the executor
                # to avoid adding another state key.
                ctx.session.state[STATE_OBJECTIVE] = step_to_execute
                async for event in self.executor.run_async(ctx):
                    yield event
                
                # Update executed steps
                ctx.session.state[STATE_EXECUTED_STEPS].append(step_to_execute)

            else:
                logger.error(f"[{self.name}] Unknown next action: {next_action}. Aborting.")
                break
        else:
            logger.warning(f"[{self.name}] Max turns ({max_turns}) reached. Aborting.")


################################################################################


async def main():
    """
    Sets up the runner and session, then runs the OrchestratorAgent.
    """
    # --- Instantiate Agents ---
    planner = build_planner_agent()
    executor = build_execution_agent()
    decider = build_decider_agent()
    orchestrator = OrchestratorAgent(planner, executor, decider)

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

    session_id = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )

    # --- Stream and Log Events ---
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=initial_message
    )

    final_response = "No final response captured."
    async for event in events:
        if event.is_final_response() and event.content:
            final_response = event.content.stringify_content()
            logger.info(f"====== FINAL RESPONSE from [{event.author}] ======\n{final_response}")

    # --- Inspect Final State ---
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    logger.info("\n--- Final Session State ---")
    print(json.dumps(final_session.state, indent=2))
    logger.info("--- Run Finished ---\n")


if __name__ == "__main__":
    asyncio.run(main())
