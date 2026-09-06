from fastapi import APIRouter, status
from fastapi.responses import JSONResponse, StreamingResponse
from api.handlers import MIDDLEWARE_ERROR_STATUS_MAP
from core.state_manager import StateManager
from core.orchestrator import Orchestrator
from core.tools import pre_processing
from core.types.contexts import GameContext, NPCContext
from api.schemas import ComposedDialogue, DialogueStreamRequest, MiddlewareStatusResponse
from api.errors import ALL_ERROR_RESPONSES, MIDDLEWARE_ERROR_RESPONSES, PREPROCESSING_ERROR_RESPONSES, ROUTING_CONFIG_ERROR_RESPONSES, error_responses
from core.types.enums import MiddlewareState
from core.tools.errors import MiddlewareError, MiddlewareErrorCode
router = APIRouter(tags=["dialogue"])

@router.get("/health", summary="Health Check", description="Returns OK if the service is running.")
def health():
    return {"status": "ok"}

@router.get(
    "/status",
    summary="Middleware Status",
    description="Returns the current middleware state.",
    responses={
        200: {"model": MiddlewareStatusResponse, "description": "Middleware is idling."},
        409: {"model": MiddlewareStatusResponse, "description": "Conflict - The middleware is busy generating a response."},
        503: {"model": MiddlewareStatusResponse, "description": "Service Unavailable - The middleware is starting up or setting context."},
    },
)
def middleware_status():
    state_manager = StateManager()
    current_state = state_manager.state

    match current_state:
        case MiddlewareState.IDLE:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"state": current_state.value},
            )
        case MiddlewareState.STARTING:
            error_code = MiddlewareErrorCode.STARTING
            message = "The middleware is starting."
        case MiddlewareState.SETTING_CONTEXT:
            error_code = MiddlewareErrorCode.SETTING_CONTEXT
            message = "The middleware is setting the game context."
        case MiddlewareState.GENERATING:
            error_code = MiddlewareErrorCode.GENERATING
            message = "The middleware is generating the dialogue."
        case _:
            error_code = None
            message = f"Unknown middleware state: {current_state.value}"

    http_status = MIDDLEWARE_ERROR_STATUS_MAP.get(
        error_code, status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "state": current_state.value,
            "error_code": error_code.value if error_code else None,
            "message": message,
        },
    )

@router.post(
	"/set-game-context",
	summary="Set Global Game Context",
	description="Sets the global game world context including environment, epoch, and world state. This should be called when the player enters a new zone or a major world event occurs.",
	responses={
              **error_responses(ROUTING_CONFIG_ERROR_RESPONSES,PREPROCESSING_ERROR_RESPONSES, MIDDLEWARE_ERROR_RESPONSES), 
              200: {"description": "Context set successfully."}
              }
)
def set_game_context(game_context: GameContext):

    if not StateManager().is_in(MiddlewareState.IDLE):
        raise MiddlewareError(code=MiddlewareErrorCode.SETTING_CONTEXT, errors=["The middleware is busy setting the game context."])

    if not Orchestrator().guardrail.validate_game_context(game_context):
        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["The middleware refused the game context."])

    game_context = pre_processing.normalize_and_validate_game_context(game_context)

    Orchestrator().set_game_context(game_context)

    return {"status": "ok"}

@router.post(
	"/generate-dialogue",
	response_model=ComposedDialogue,
	summary="Generate NPC Dialogue",
	description="Generates dialogue and available player responses based on the provided NPC context and current intent.",
	responses={**ALL_ERROR_RESPONSES, 200: {"description": "Dialogue generated successfully."}}
)
def generate_dialogue(npc_context: NPCContext):

    if Orchestrator().game_context is None:
        raise MiddlewareError(code=MiddlewareErrorCode.CONTEXT_NOT_SET, errors=["Game context is not set."])
       
    if not StateManager().is_in(MiddlewareState.IDLE):
        raise MiddlewareError(code=MiddlewareErrorCode.GENERATING, errors=["The middleware is busy generating."])

    if not Orchestrator().guardrail.validate_npc_context(npc_context):
            raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["The middleware refused the npc context."])
    
    npc_context = pre_processing.normalize_and_validate_npc_context(npc_context)
    
    dialogue: ComposedDialogue = Orchestrator().generate_dialogue(npc_context, None)

    if dialogue is None:
        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["Dialogue generation was refused."])

    return dialogue

@router.post(
    "/start-dialogue-stream",
    response_class=StreamingResponse,
    summary="Starts NPC Dialogue using streaming mode",
    description="Streams the generated dialogue line by line to reduce perceived latency for the player, cleaning the dialogue history.",
    responses={**ALL_ERROR_RESPONSES, 200: {"description": "Stream of dialogue text."}},
)
def start_dialogue_stream(npc_context: NPCContext):
    if Orchestrator().game_context is None:
        raise MiddlewareError(code=MiddlewareErrorCode.CONTEXT_NOT_SET, errors=["Game context is not set."])

    if not StateManager().is_in(MiddlewareState.IDLE):
        raise MiddlewareError(code=MiddlewareErrorCode.GENERATING, errors=["The middleware is busy generating."])

    if not Orchestrator().guardrail.validate_npc_context(npc_context):
        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["The middleware refused the npc context."])

    npc_context = pre_processing.normalize_and_validate_npc_context(npc_context)

    Orchestrator().dialogue_history.clear_dialogue_history()

    stream = Orchestrator().generate_dialogue_stream(npc_context, None)

    return StreamingResponse(stream, media_type="text/plain")


@router.post(
    "/continue-dialogue-stream",
    response_class=StreamingResponse,
    summary="Continues the NPC Dialogue using streaming mode",
    description="Streams the generated dialogue line by line to reduce perceived latency for the player, without cleaning the dialogue history.",
    responses={
        **ALL_ERROR_RESPONSES,
        200: {"description": "Stream of dialogue text."},
    },
)
def continue_dialogue_stream(request: DialogueStreamRequest):
    if Orchestrator().game_context is None:
        raise MiddlewareError(code=MiddlewareErrorCode.CONTEXT_NOT_SET, errors=["Game context is not set."])

    if Orchestrator().dialogue_history.is_empty():
        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["Dialogue history is empty."])

    if not StateManager().is_in(MiddlewareState.IDLE):
        raise MiddlewareError(code=MiddlewareErrorCode.GENERATING, errors=["The middleware is busy generating."])

    npc_context = request.npc_context

    if not Orchestrator().guardrail.validate_npc_context(npc_context):
        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["The middleware refused the npc context."])

    npc_context = pre_processing.normalize_and_validate_npc_context(npc_context)

    stream = Orchestrator().generate_dialogue_stream(npc_context, request.last_player_choice)

    return StreamingResponse(stream, media_type="text/plain")