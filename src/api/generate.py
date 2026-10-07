from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from api.dependencies import get_generate_service
from api.handlers import MIDDLEWARE_ERROR_STATUS_MAP
from core.generation.generate_service import GenerateService
from core.infrastructure.state_manager import StateManager
from core.types.contexts import GameContext
from api.schemas import ComposedDialogue, GenerateDialogueRequest, MiddlewareStatusResponse
from api.errors import ALL_ERROR_RESPONSES, MIDDLEWARE_ERROR_RESPONSES, PREPROCESSING_ERROR_RESPONSES, ROUTING_CONFIG_ERROR_RESPONSES, error_responses
from core.types.dataclasses import DialogueStream
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
def set_game_context(game_context: GameContext, service: GenerateService = Depends(get_generate_service)):
    service.set_game_context(game_context)
    return {"status": "ok"}

@router.post(
    "/generate-dialogue",
    response_model=None,
    summary="Generate NPC Dialogue",
    description=(
        "Generates the NPC dialogue and the available player responses. "
        "If `last_player_choice` is omitted a new dialogue is started (history cleared), "
        "otherwise the dialogue is continued. With `stream=true` the dialogue is streamed as plain text."
    ),
    responses={
        **ALL_ERROR_RESPONSES,
        200: {
            "description": "Dialogue generated successfully.",
            "content": {
                "application/json": {"schema": ComposedDialogue.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
        },
    },
)
def generate_dialogue(request: GenerateDialogueRequest, service: GenerateService = Depends(get_generate_service)) -> ComposedDialogue | StreamingResponse:

    if request.stream:
        dialogue_stream: DialogueStream = service.generate_dialogue_stream(
            request.npc_context, request.last_player_choice
        )
        return StreamingResponse(
            dialogue_stream.chunks,
            media_type="text/plain",
        )

    dialogue = service.generate_dialogue(request.npc_context, request.last_player_choice)

    if dialogue is None:
        raise MiddlewareError(
            code=MiddlewareErrorCode.REFUSED,
            errors=["Dialogue generation was refused."],
        )

    return dialogue