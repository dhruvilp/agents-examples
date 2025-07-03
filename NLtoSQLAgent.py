import asyncio
import logging
import json
from typing import AsyncGenerator, Dict, Any

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
APP_NAME = "SqlGeneratorApp"
USER_ID = "sql_user_789"

# --- State Keys for Session Management ---
STATE_OBJECTIVE = "objective"
STATE_PLAN = "plan"
STATE_EXECUTED_STEPS = "executed_steps"
STATE_LAST_EXECUTION_RESULT = "last_execution_result"
STATE_DECISION = "decision"
STATE_CURRENT_STEP = "current_step"
STATE_DB_SCHEMA = "db_schema"
STATE_GENERATED_SQL = "generated_sql"


# --- Tool Definitions for SQL Generation ---

def get_database_schema() -> Dict[str, str]:
    """
    Retrieves the schema of the database.

    In a real application, this would connect to the database to fetch metadata.
    For this example, we return a hardcoded mock schema.
    """
    logger.info("TOOL CALL: get_database_schema()")
    schema = """
    Table: customers
      - customer_id (INTEGER, PRIMARY KEY)
      - first_name (TEXT)
      - last_name (TEXT)
      - email (TEXT)
      - city (TEXT)
      - country (TEXT)

    Table: orders
      - order_id (INTEGER, PRIMARY KEY)
      - customer_id (INTEGER, FOREIGN KEY to customers.customer_id)
      - order_date (DATE)
      - amount (DECIMAL)
    """
    return {"schema": schema.strip()}


def validate_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Validates the syntax of a given SQL query.

    In a real application, this might run `EXPLAIN` or `DRY RUN`.
    For this example, we perform basic string checks.
    """
    logger.info(f"TOOL CALL: validate_sql_query(sql_query='{sql_query}')")
    query_lower = sql_query.lower()
    if "select" not in query_lower or "from" not in query_lower:
        return {
            "is_valid": False,
            "error": "Syntax Error: Query must contain SELECT and FROM clauses.",
        }
    # A simple check for a common mistake.
    if "from customers" in query_lower and "name" in query_lower and "first_name" not in query_lower and "last_name" not in query_lower:
        return {
            "is_valid": False,
            "error": "Validation Error: The 'customers' table has 'first_name' and 'last_name' columns, not a 'name' column. Please correct the query.",
        }
    return {"is_valid": True, "validated_query": sql_query}


# --- Sub-Agent Class Definitions ---


class SqlPlannerAgent(LlmAgent):
    """
    Creates a plan to generate and validate a SQL query from a natural language question.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="SqlPlannerAgent",
            model=model,
            instruction=f"""
            You are an expert planner specializing in database query generation.
            Your task is to create a robust, step-by-step plan to convert a natural
            language question into a valid SQL query.

            **Available Tools for the Execution Agent:**
            - `get_database_schema()`: Fetches the structure of the database.
            - `validate_sql_query(sql_query: str)`: Checks the generated SQL for correctness.

            **Task:**
            Create a JSON object with a "plan" array. The plan must include these logical steps:
            1.  Get the database schema to understand the table structure.
            2.  Draft a SQL query based on the user's objective and the schema.
            3.  Validate the drafted SQL query using the provided tool.
            4.  Present the final, validated SQL query as the result.

            **Input:**
            - Objective: {{{STATE_OBJECTIVE}}}

            **Example Output:**
            {{
              "plan": [
                "Use the get_database_schema tool to understand the available tables and columns.",
                "Draft a SQL query that answers the user's question, making sure to use the correct table and column names from the schema.",
                "Use the validate_sql_query tool on the drafted query to ensure it is syntactically correct.",
                "Output the final, validated SQL query."
              ]
            }}
            """,
            output_key=STATE_PLAN,
            response_mime_type="application/json",
        )


class SqlExecutionAgent(LlmAgent):
    """
    Executes a single step of the SQL generation plan. It can draft queries and use tools.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="SqlExecutionAgent",
            model=model,
            tools=[
                FunctionTool.from_function(get_database_schema),
                FunctionTool.from_function(validate_sql_query),
            ],
            instruction=f"""
            You are a world-class SQL developer. Your role is to execute a single step
            from a plan to generate a SQL query. Pay close attention to the database schema.

            **Task:**
            - Read the 'current_step' you have been given.
            - If the step requires a tool (like `get_database_schema` or `validate_sql_query`), call it.
            - If the step requires you to draft a query, write the SQL code.
            - Your output should be the direct result of the step's execution.

            **Context from Previous Steps (if available):**
            - Database Schema: {{{STATE_DB_SCHEMA}}}
            - Generated SQL (for validation step): {{{STATE_GENERATED_SQL}}}
            - User's Objective: {{{STATE_OBJECTIVE}}}

            **Current Step to Execute:** {{{STATE_CURRENT_STEP}}}
            """,
            output_key=STATE_LAST_EXECUTION_RESULT,
        )


class CoordinatorAgent(LlmAgent):
    """
    Coordinates the SQL generation workflow by analyzing progress and deciding the next action.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(
            name="CoordinatorAgent",
            model=model,
            instruction=f"""
            You are the coordinator for a Natural Language to SQL generation task.
            Your role is to analyze the progress and decide the next action based on the plan and execution results.

            **Decision Logic:**
            1.  If there is no plan yet, set 'next_action' to 'PLAN'.
            2.  If the last step was a validation that returned `is_valid: true`, the task is complete. Set 'next_action' to 'FINISH' and 'is_done' to true.
            3.  If the last step was a validation that returned `is_valid: false`, set 'next_action' to 'REPLAN' so a new query can be drafted.
            4.  If the plan is not yet fully executed, find the next step. Set 'next_action' to 'EXECUTE' and provide the 'next_step'.

            **Input State:**
            - Objective: {{{STATE_OBJECTIVE}}}
            - Plan: {{{STATE_PLAN}}}
            - Executed Steps: {{{STATE_EXECUTED_STEPS}}}
            - Last Execution Result: {{{STATE_LAST_EXECUTION_RESULT}}}

            Respond with a JSON object containing your decision.
            """,
            output_key=STATE_DECISION,
            response_mime_type="application/json",
        )


class OrchestratorAgent(BaseAgent):
    """
    Manages the workflow of planning and execution to generate a SQL query.
    """
    model_config = {"arbitrary_types_allowed": True}

    planner: SqlPlannerAgent
    executor: SqlExecutionAgent
    coordinator: CoordinatorAgent

    def __init__(
        self,
        planner: SqlPlannerAgent,
        executor: SqlExecutionAgent,
        coordinator: CoordinatorAgent,
    ):
        super().__init__(
            name="OrchestratorAgent",
            sub_agents=[planner, executor, coordinator],
            planner=planner,
            executor=executor,
            coordinator=coordinator,
        )

    def _update_context(self, ctx: InvocationContext):
        """Helper to update context between steps."""
        last_result_str = ctx.session.state.get(STATE_LAST_EXECUTION_RESULT, "{}")
        try:
            last_result = json.loads(last_result_str)
            if "schema" in last_result:
                ctx.session.state[STATE_DB_SCHEMA] = last_result["schema"]
            if "sql_query" in last_result:
                ctx.session.state[STATE_GENERATED_SQL] = last_result["sql_query"]
            if "validated_query" in last_result:
                ctx.session.state[STATE_GENERATED_SQL] = last_result["validated_query"]
        except (json.JSONDecodeError, TypeError):
            # It's okay if the result is not JSON, it might be a simple string.
            pass


    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting SQL Generation Workflow.")

        if ctx.new_message:
            ctx.session.state[STATE_OBJECTIVE] = ctx.new_message.stringify_content()
        ctx.session.state.setdefault(STATE_PLAN, None)
        ctx.session.state.setdefault(STATE_EXECUTED_STEPS, [])
        ctx.session.state.setdefault(STATE_LAST_EXECUTION_RESULT, None)
        ctx.session.state.setdefault(STATE_DB_SCHEMA, "")
        ctx.session.state.setdefault(STATE_GENERATED_SQL, "")

        max_turns = 10
        for turn_count in range(max_turns):
            logger.info(f"--- Turn {turn_count + 1}/{max_turns} ---")

            # 1. COORDINATE the next action
            logger.info(f"[{self.name}] Running CoordinatorAgent...")
            async for event in self.coordinator.run_async(ctx):
                yield event

            decision = json.loads(ctx.session.state.get(STATE_DECISION, "{}"))
            logger.info(f"[{self.name}] Coordinator Decision: {decision.get('thought', 'No thought provided.')}")

            if decision.get("is_done"):
                logger.info(f"[{self.name}] Objective achieved. Workflow finished.")
                final_sql = ctx.session.state.get(STATE_GENERATED_SQL, "No SQL generated.")
                yield Event.from_final_response(author=self.name, content=final_sql)
                break

            next_action = decision.get("next_action")

            # 2. ACT based on the decision
            if next_action in ("PLAN", "REPLAN"):
                logger.info(f"[{self.name}] Running SqlPlannerAgent to {next_action.lower()}...")
                if next_action == "REPLAN":
                    ctx.session.state[STATE_EXECUTED_STEPS] = [] # Reset for new plan
                async for event in self.planner.run_async(ctx):
                    yield event

            elif next_action == "EXECUTE":
                step_to_execute = decision.get("next_step")
                if not step_to_execute:
                    logger.error("Coordinator chose EXECUTE but provided no step.")
                    break
                
                logger.info(f"[{self.name}] Running SqlExecutionAgent for step: '{step_to_execute}'")
                ctx.session.state[STATE_CURRENT_STEP] = step_to_execute
                async for event in self.executor.run_async(ctx):
                    yield event
                
                ctx.session.state[STATE_EXECUTED_STEPS].append(step_to_execute)
                self._update_context(ctx) # Update context with results of the execution

            else:
                logger.error(f"[{self.name}] Unknown next action: '{next_action}'. Aborting.")
                break
        else:
            logger.warning(f"[{self.name}] Max turns ({max_turns}) reached. Aborting.")


async def main():
    """
    Sets up and runs the Natural Language to SQL Query Generator Agent.
    """
    # --- Instantiate Agents ---
    planner = SqlPlannerAgent()
    executor = SqlExecutionAgent()
    coordinator = CoordinatorAgent()
    orchestrator = OrchestratorAgent(planner, executor, coordinator)

    # --- Setup Runner ---
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Define the Objective and Run ---
    objective = "Show me the first name and email of all customers who live in London."
    logger.info(f"--- Starting New Run ---")
    logger.info(f"Objective: {objective}")

    initial_message = genai_types.Content(
        role="user", parts=[genai_types.Part(text=objective)]
    )
    session_id = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    # --- Stream Events and Print Final Response ---
    final_response = "No final response was captured."
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=initial_message
    ):
        if event.is_final_response() and event.content:
            final_response = event.content.stringify_content()
            print("\n" + "="*50)
            print("✅ Final Generated SQL Query:")
            print(final_response)
            print("="*50 + "\n")
        elif event.is_error():
            logger.error(f"====== ERROR from [{event.author}] ======\n{event.error_message}")

    # --- Optional: Inspect Final State ---
    # final_session = await session_service.get_session(
    #     app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    # )
    # logger.info("\n--- Final Session State ---")
    # print(json.dumps(json.loads(final_session.model_dump_json()), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
