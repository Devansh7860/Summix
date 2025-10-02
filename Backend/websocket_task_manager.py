"""
WebSocket-based Task Management System
Real-time task tracking and cancellation for YouTube Assistant
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Set, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(Enum):
    VIDEO_SUMMARY = "video_summary"
    PLAYLIST_SUMMARY = "playlist_summary"
    QUESTION_ANSWER = "question_answer"

@dataclass
class Task:
    task_id: str
    user_id: str
    task_type: TaskType
    content_id: str  # video_id or playlist_id
    status: TaskStatus
    created_at: float
    updated_at: float
    progress: int = 0  # 0-100%
    message: str = ""
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            **asdict(self),
            'task_type': self.task_type.value,
            'status': self.status.value
        }

class WebSocketTaskManager:
    def __init__(self):
        # Active tasks by task_id
        self.tasks: Dict[str, Task] = {}
        
        # WebSocket connections by user_id
        self.connections: Dict[str, WebSocket] = {}
        
        # User tasks mapping
        self.user_tasks: Dict[str, Set[str]] = {}
        
        # Task locks for thread safety
        self._task_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def connect_user(self, websocket: WebSocket, user_id: str):
        """Connect a user's WebSocket"""
        await websocket.accept()
        
        async with self._global_lock:
            # Disconnect existing connection if any
            if user_id in self.connections:
                try:
                    await self.connections[user_id].close()
                except:
                    pass
            
            self.connections[user_id] = websocket
            logger.info(f"✅ User {user_id} connected via WebSocket")
            
            # Send current tasks to user
            await self._send_user_tasks(user_id)

    async def disconnect_user(self, user_id: str):
        """Disconnect a user's WebSocket"""
        async with self._global_lock:
            if user_id in self.connections:
                try:
                    await self.connections[user_id].close()
                except:
                    pass
                del self.connections[user_id]
                
            logger.info(f"🔌 User {user_id} disconnected")
            
            # Cancel user tasks immediately when they disconnect
            # The frontend should handle cancellation when sidepanel closes
            cancelled_count = await self._cancel_user_tasks(user_id)
            if cancelled_count > 0:
                logger.info(f"🚫 Cancelled {cancelled_count} tasks for disconnected user {user_id}")
            else:
                logger.info(f"ℹ️ No active tasks found for user {user_id}")

    async def _handle_delayed_disconnect_cleanup(self, user_id: str, delay_seconds: int = 10):
        """Handle cleanup after user disconnect with delay to avoid cancelling on temporary disconnects"""
        await asyncio.sleep(delay_seconds)
        
        # Check if user reconnected
        async with self._global_lock:
            if user_id in self.connections:
                logger.info(f"🔄 User {user_id} reconnected - skipping task cancellation")
                return
        
        # User still disconnected after delay - cancel their tasks
        logger.info(f"� User {user_id} permanently disconnected - cancelling all tasks")
        cancelled_count = await self._cancel_user_tasks(user_id)
        if cancelled_count > 0:
            logger.info(f"📋 Cancelled {cancelled_count} tasks for disconnected user {user_id}")

    async def create_task(self, user_id: str, task_type: TaskType, content_id: str) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        current_time = time.time()
        
        logger.info(f"🆕 Creating new task {task_id} for user {user_id}, content {content_id}")
        
        # Cancel any existing tasks for this user with same content
        # This prevents duplicate processing when user switches content
        await self._cancel_existing_tasks(user_id, content_id)
        
        task = Task(
            task_id=task_id,
            user_id=user_id,
            task_type=task_type,
            content_id=content_id,
            status=TaskStatus.PENDING,
            created_at=current_time,
            updated_at=current_time,
            message="Task created"
        )
        
        async with self._global_lock:
            self.tasks[task_id] = task
            
            # Add to user tasks
            if user_id not in self.user_tasks:
                self.user_tasks[user_id] = set()
            self.user_tasks[user_id].add(task_id)
            
            # Create task lock
            self._task_locks[task_id] = asyncio.Lock()
        
        await self._broadcast_task_update(task)
        logger.info(f"✅ Created task {task_id} for user {user_id}, status: {task.status}")
        return task_id

    async def update_task(self, task_id: str, status: Optional[TaskStatus] = None, 
                         progress: Optional[int] = None, message: Optional[str] = None,
                         result: Optional[str] = None, error: Optional[str] = None):
        """Update a task's status and broadcast to client"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Get or create task lock
        if task_id not in self._task_locks:
            self._task_locks[task_id] = asyncio.Lock()
        
        async with self._task_locks[task_id]:
            if status:
                task.status = status
            if progress is not None:
                task.progress = progress
            if message:
                task.message = message
            if result:
                task.result = result
            if error:
                task.error = error
            
            task.updated_at = time.time()
        
        await self._broadcast_task_update(task)
        logger.info(f"🔄 Updated task {task_id}: {task.status.value} ({task.progress}%)")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        import traceback
        stack_trace = ''.join(traceback.format_stack())
        logger.info(f"🚫 CANCEL TASK CALLED for {task_id}")
        logger.info(f"📍 STACK TRACE:\n{stack_trace}")
        
        if task_id not in self.tasks:
            logger.warning(f"⚠️ Attempted to cancel non-existent task: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            logger.info(f"⏰ Task {task_id} already in final state: {task.status}")
            return False
        
        logger.info(f"❌ Cancelling task {task_id} (current status: {task.status})")
        await self.update_task(task_id, 
                             status=TaskStatus.CANCELLED, 
                             message="")  # Empty message to avoid showing cancellation in UI
        
        logger.info(f"✅ Task {task_id} marked as cancelled")
        return True

    async def cancel_user_tasks(self, user_id: str, content_id: Optional[str] = None) -> int:
        """Cancel all tasks for a user (optionally for specific content)"""
        return await self._cancel_user_tasks(user_id, content_id)

    async def is_task_cancelled(self, task_id: str) -> bool:
        """Check if a task is cancelled"""
        if task_id not in self.tasks:
            return False
        return self.tasks[task_id].status == TaskStatus.CANCELLED

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    async def get_user_tasks(self, user_id: str) -> List[Task]:
        """Get all tasks for a user"""
        if user_id not in self.user_tasks:
            return []
        
        user_task_ids = self.user_tasks[user_id]
        return [self.tasks[task_id] for task_id in user_task_ids if task_id in self.tasks]

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the current status of a task"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_task_status_sync(self, task_id: str) -> Optional[TaskStatus]:
        """Synchronous version of get_task_status for non-async contexts"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None

    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        
        async with self._global_lock:
            for task_id, task in self.tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    current_time - task.updated_at > max_age_seconds):
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            await self._remove_task(task_id)
        
        logger.info(f"🧹 Cleaned up {len(to_remove)} old tasks")
        return len(to_remove)

    # Private methods
    async def _cancel_existing_tasks(self, user_id: str, content_id: str):
        """Cancel existing tasks for user with same content"""
        logger.info(f"🔍 Checking for existing tasks to cancel for user {user_id}, content {content_id}")
        
        if user_id not in self.user_tasks:
            logger.info(f"ℹ️ No existing tasks found for user {user_id}")
            return
        
        cancelled_count = 0
        for task_id in list(self.user_tasks[user_id]):
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if (task.content_id == content_id and 
                    task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]):
                    logger.info(f"🔄 Cancelling existing task {task_id} for content {content_id}")
                    await self.cancel_task(task_id)
                    cancelled_count += 1
        
        if cancelled_count > 0:
            logger.info(f"📋 Cancelled {cancelled_count} existing tasks for user {user_id}, content {content_id}")

    async def _cancel_user_tasks(self, user_id: str, content_id: Optional[str] = None) -> int:
        """Internal method to cancel user tasks"""
        if user_id not in self.user_tasks:
            return 0
        
        cancelled_count = 0
        for task_id in list(self.user_tasks[user_id]):
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if (content_id is None or task.content_id == content_id):
                    if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                        await self.cancel_task(task_id)
                        cancelled_count += 1
        
        return cancelled_count

    async def _broadcast_task_update(self, task: Task):
        """Broadcast task update to the user's WebSocket"""
        user_id = task.user_id
        
        if user_id in self.connections:
            try:
                message = {
                    "type": "task_update",
                    "task": task.to_dict()
                }
                await self.connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send task update to {user_id}: {e}")
                # Remove broken connection
                await self.disconnect_user(user_id)

    async def _send_user_tasks(self, user_id: str):
        """Send all user tasks to their WebSocket"""
        if user_id not in self.connections:
            return
        
        tasks = await self.get_user_tasks(user_id)
        
        try:
            message = {
                "type": "tasks_list",
                "tasks": [task.to_dict() for task in tasks]
            }
            await self.connections[user_id].send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send tasks list to {user_id}: {e}")

    async def _remove_task(self, task_id: str):
        """Remove a task completely"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        user_id = task.user_id
        
        async with self._global_lock:
            # Remove from tasks
            del self.tasks[task_id]
            
            # Remove from user tasks
            if user_id in self.user_tasks:
                self.user_tasks[user_id].discard(task_id)
                if not self.user_tasks[user_id]:
                    del self.user_tasks[user_id]
            
            # Remove task lock
            if task_id in self._task_locks:
                del self._task_locks[task_id]

# Global task manager instance
task_manager = WebSocketTaskManager()

# Cleanup task (run periodically)
async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        try:
            await task_manager.cleanup_completed_tasks()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(60)  # Retry in 1 minute