import os
import json
import datetime
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

class ChatSessionManager:
    """
    Manages chat sessions for the OpenRouter node.
    Handles creating, loading, and updating chat conversations with automatic session management.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the chat session manager.
        
        Args:
            base_path: Base directory for storing chats. Defaults to node's directory + /chats
        """
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "chats")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.current_session_path = None
        self.session_timeout_hours = 1  # Sessions expire after 1 hour of inactivity
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use in filenames.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length of the sanitized text
            
        Returns:
            Sanitized text safe for filenames
        """
        # Get first 5 words or max_length characters, whichever is shorter
        words = text.split()[:5]
        text = " ".join(words)
        
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        
        # Truncate to max length
        if len(text) > max_length:
            text = text[:max_length].rstrip('_')
        
        return text.lower()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in format YYYYMMDD_HHMMSS"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_iso_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.datetime.now().isoformat()
    
    def _find_active_session(self) -> Optional[Path]:
        """
        Find an active session within the timeout period.
        
        Returns:
            Path to active session directory or None
        """
        if not self.base_path.exists():
            return None
        
        current_time = time.time()
        
        # Get all session directories
        sessions = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        # Sort by modification time (most recent first)
        sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for session_dir in sessions:
            conversation_file = session_dir / "conversation.json"
            if conversation_file.exists():
                # Check if session is within timeout period
                last_modified = conversation_file.stat().st_mtime
                hours_elapsed = (current_time - last_modified) / 3600
                
                if hours_elapsed <= self.session_timeout_hours:
                    return session_dir
        
        return None
    
    def _create_new_session(self, first_message: str) -> Path:
        """
        Create a new chat session directory.
        
        Args:
            first_message: The first user message to use in the session name
            
        Returns:
            Path to the new session directory
        """
        timestamp = self._get_timestamp()
        sanitized_message = self._sanitize_filename(first_message)
        
        # Create session directory name
        session_name = f"session_{timestamp}_{sanitized_message}"
        session_path = self.base_path / session_name
        
        # Handle potential naming conflicts
        counter = 1
        while session_path.exists():
            session_path = self.base_path / f"{session_name}_{counter}"
            counter += 1
        
        session_path.mkdir(exist_ok=True)
        return session_path
    
    def get_or_create_session(self, user_message: str, system_prompt: str) -> Tuple[Path, List[Dict]]:
        """
        Get an active session or create a new one.
        
        Args:
            user_message: The current user message
            system_prompt: The system prompt to use
            
        Returns:
            Tuple of (session_path, message_history)
        """
        # Try to find an active session
        active_session = self._find_active_session()
        
        if active_session:
            self.current_session_path = active_session
            messages = self.load_conversation(active_session)
            return active_session, messages
        else:
            # Create new session
            new_session = self._create_new_session(user_message)
            self.current_session_path = new_session
            
            # Initialize with system prompt
            messages = [{"role": "system", "content": system_prompt}]
            
            # Create initial conversation file
            conversation_data = {
                "session_id": new_session.name,
                "created_at": self._get_iso_timestamp(),
                "last_updated": self._get_iso_timestamp(),
                "messages": messages
            }
            
            conversation_file = new_session / "conversation.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            return new_session, messages
    
    def load_conversation(self, session_path: Path) -> List[Dict]:
        """
        Load conversation history from a session.
        
        Args:
            session_path: Path to the session directory
            
        Returns:
            List of messages in the conversation
        """
        conversation_file = session_path / "conversation.json"
        
        if not conversation_file.exists():
            return []
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading conversation from {conversation_file}: {e}")
            return []
    
    def save_conversation(self, session_path: Path, messages: List[Dict]):
        """
        Save conversation history to a session.
        
        Args:
            session_path: Path to the session directory
            messages: List of messages to save
        """
        conversation_file = session_path / "conversation.json"
        
        # Load existing data to preserve metadata
        existing_data = {}
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Update conversation data
        conversation_data = {
            "session_id": existing_data.get("session_id", session_path.name),
            "created_at": existing_data.get("created_at", self._get_iso_timestamp()),
            "last_updated": self._get_iso_timestamp(),
            "messages": messages
        }
        
        # Save to file
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving conversation to {conversation_file}: {e}")
    
    def append_message(self, role: str, content: str, session_path: Optional[Path] = None):
        """
        Append a message to the current or specified session.
        
        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
            session_path: Optional specific session path, uses current if not provided
        """
        if session_path is None:
            session_path = self.current_session_path
        
        if session_path is None:
            raise ValueError("No active session to append message to")
        
        # Load existing messages
        messages = self.load_conversation(session_path)
        
        # Append new message
        messages.append({
            "role": role,
            "content": content,
            "timestamp": self._get_iso_timestamp()
        })
        
        # Save updated conversation
        self.save_conversation(session_path, messages)
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get a list of recent chat sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata dictionaries
        """
        if not self.base_path.exists():
            return []
        
        sessions = []
        session_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        # Sort by modification time (most recent first)
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for session_dir in session_dirs[:limit]:
            conversation_file = session_dir / "conversation.json"
            if conversation_file.exists():
                try:
                    with open(conversation_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data.get("session_id", session_dir.name),
                            "path": str(session_dir),
                            "created_at": data.get("created_at"),
                            "last_updated": data.get("last_updated"),
                            "message_count": len(data.get("messages", [])),
                            "first_user_message": next(
                                (msg["content"] for msg in data.get("messages", []) 
                                 if msg["role"] == "user"), 
                                "No user message"
                            )[:100]  # First 100 chars of first user message
                        })
                except (json.JSONDecodeError, IOError):
                    continue
        
        return sessions
    
    def clear_old_sessions(self, days: int = 30):
        """
        Clear sessions older than specified days.
        
        Args:
            days: Number of days to keep sessions
        """
        if not self.base_path.exists():
            return
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 3600)
        
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                # Check modification time
                if session_dir.stat().st_mtime < cutoff_time:
                    # Remove old session
                    try:
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"Removed old session: {session_dir.name}")
                    except Exception as e:
                        print(f"Error removing session {session_dir.name}: {e}")