"""
API Module for PropertyAssistAI
------------------------------
This module implements the FastAPI endpoints for the PropertyAssistAI service,
optimized for high performance and low latency.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.auth import get_current_user, User
from app.core.config import settings
from app.core.conversation import ConversationHandler
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    WebhookRequest,
    WebhookResponse,
    HealthCheckResponse,
)
from app.utils.metrics import MetricsCollector


# Initialize router
router = APIRouter()

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = MetricsCollector()


@router.get("/health", response_model=HealthCheckResponse, tags=["system"])
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring systems.
    
    Returns:
        Health status information
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT
    )


@router.post("/chat", response_model=ChatResponse, tags=["conversation"])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    conversation_handler: ConversationHandler = Depends(),
    current_user: Optional[User] = Depends(get_current_user),
) -> ChatResponse:
    """
    Process a chat message and generate a response.
    
    Args:
        request: Chat request containing the message
        background_tasks: FastAPI background tasks
        conversation_handler: Injected conversation handler
        current_user: Authenticated user (optional)
        
    Returns:
        ChatResponse with AI-generated reply
    """
    # Start timing for performance monitoring
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log incoming request
    logger.info(
        f"Chat request {request_id} received from "
        f"user {current_user.id if current_user else 'anonymous'}"
    )
    
    # Track metrics
    metrics.increment("chat_requests")
    
    # Create or retrieve conversation ID
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    try:
        # Process the message with timeout handling
        response = await conversation_handler.process_message(
            message=request.message,
            conversation_id=conversation_id,
            background_tasks=background_tasks
        )
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log success and timing
        logger.info(
            f"Chat request {request_id} processed in {process_time:.3f}s"
        )
        
        # Track timing metrics
        metrics.record("chat_processing_time", process_time)
        
        # Construct response
        return ChatResponse(
            conversation_id=conversation_id,
            message=response.content,
            processed_time=process_time,
            request_id=request_id,
            lead_created=response.lead_created,
            detected_intents=response.processed_intents
        )
        
    except Exception as e:
        # Log error
        logger.error(f"Error processing chat request {request_id}: {str(e)}")
        
        # Track error metrics
        metrics.increment("chat_errors")
        
        # Return error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )


@router.post("/webhook", response_model=WebhookResponse, tags=["integrations"])
async def webhook(
    request: WebhookRequest,
    background_tasks: BackgroundTasks,
    conversation_handler: ConversationHandler = Depends(),
    current_user: User = Depends(get_current_user),
) -> WebhookResponse:
    """
    Process CRM webhook events.
    
    Args:
        request: Webhook request with event data
        background_tasks: FastAPI background tasks
        conversation_handler: Injected conversation handler
        current_user: Authenticated user
        
    Returns:
        Webhook processing status
    """
    # Validate API key
    if not current_user.has_permission("webhook:receive"):
        logger.warning(f"Unauthorized webhook attempt from {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access webhooks"
        )
    
    # Log webhook event
    logger.info(f"Webhook received: {request.event_type} from {request.source}")
    
    # Track metrics
    metrics.increment("webhooks_received")
    metrics.increment(f"webhook_{request.event_type}")
    
    try:
        # Process webhook in background to return response quickly
        background_tasks.add_task(
            _process_webhook,
            request,
            conversation_handler
        )
        
        return WebhookResponse(
            status="accepted",
            message="Webhook queued for processing"
        )
        
    except Exception as e:
        logger.error(f"Error queueing webhook: {str(e)}")
        metrics.increment("webhook_errors")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


@router.get("/metrics", tags=["system"])
async def get_metrics(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get system metrics (requires admin privileges).
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Dictionary of system metrics
    """
    # Check admin permissions
    if not current_user.has_permission("metrics:view"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view metrics"
        )
    
    # Get current metrics
    return metrics.get_all()


@router.post("/feedback", tags=["conversation"])
async def submit_feedback(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user),
) -> JSONResponse:
    """
    Submit user feedback for conversation quality.
    
    Args:
        request: Feedback data
        background_tasks: FastAPI background tasks
        current_user: Authenticated user (optional)
        
    Returns:
        Confirmation response
    """
    # Validate request
    if "conversation_id" not in request or "rating" not in request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields"
        )
    
    # Log feedback
    logger.info(
        f"Feedback received for conversation {request['conversation_id']}: "
        f"rating={request['rating']}"
    )
    
    # Track metrics
    metrics.increment("feedback_received")
    metrics.record("feedback_rating", float(request["rating"]))
    
    # Process feedback in background
    background_tasks.add_task(_process_feedback, request)
    
    return JSONResponse(
        content={"status": "success", "message": "Feedback submitted"},
        status_code=status.HTTP_200_OK
    )


# Background processing functions

async def _process_webhook(
    request: WebhookRequest,
    conversation_handler: ConversationHandler
) -> None:
    """
    Process webhook events in background.
    
    Args:
        request: Webhook data
        conversation_handler: Conversation handler
    """
    try:
        # Implementation depends on webhook type
        if request.event_type == "lead_updated":
            # Update conversation context with new lead data
            await conversation_handler.update_lead_context(
                lead_id=request.data.get("lead_id"),
                updates=request.data.get("updates", {})
            )
        elif request.event_type == "property_updated":
            # Update property knowledge in RAG system
            await conversation_handler.update_property_knowledge(
                property_id=request.data.get("property_id"),
                updates=request.data.get("updates", {})
            )
        
        logger.info(f"Webhook {request.event_type} processed successfully")
        metrics.increment("webhooks_processed")
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        metrics.increment("webhook_processing_errors")


async def _process_feedback(feedback: Dict[str, Any]) -> None:
    """
    Process user feedback in background.
    
    Args:
        feedback: Feedback data
    """
    try:
        # Save feedback to database
        # This would connect to your feedback storage system
        
        # If feedback is negative, flag for review
        if feedback.get("rating", 5) < 3:
            logger.warning(
                f"Negative feedback for conversation {feedback['conversation_id']}: "
                f"{feedback.get('comment', 'No comment provided')}"
            )
            metrics.increment("negative_feedback")
        
        logger.info(f"Feedback for conversation {feedback['conversation_id']} processed")
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        metrics.increment("feedback_processing_errors")