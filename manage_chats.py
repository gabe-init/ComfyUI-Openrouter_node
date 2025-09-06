#!/usr/bin/env python3
"""
Chat Session Management Utility for OpenRouter Node

This script provides utilities to manage chat sessions:
- List all chat sessions
- View a specific session
- Export sessions to different formats
- Clean up old sessions
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from chat_manager import ChatSessionManager

def list_sessions(manager: ChatSessionManager, limit: int = None):
    """List all chat sessions"""
    sessions = manager.get_recent_sessions(limit or 1000)
    
    if not sessions:
        print("No chat sessions found.")
        return
    
    print(f"\nFound {len(sessions)} chat session(s):\n")
    print(f"{'#':<4} {'Session ID':<50} {'Created':<20} {'Messages':<10} {'First Message':<50}")
    print("-" * 140)
    
    for i, session in enumerate(sessions, 1):
        created = session.get('created_at', 'Unknown')
        if created != 'Unknown':
            try:
                dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                created = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        print(f"{i:<4} {session['session_id'][:50]:<50} {created:<20} {session['message_count']:<10} {session['first_user_message'][:50]:<50}")

def view_session(manager: ChatSessionManager, session_id: str):
    """View a specific chat session"""
    session_path = manager.base_path / session_id
    
    if not session_path.exists():
        print(f"Error: Session '{session_id}' not found.")
        return
    
    messages = manager.load_conversation(session_path)
    
    if not messages:
        print("No messages found in this session.")
        return
    
    print(f"\n=== Chat Session: {session_id} ===\n")
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        timestamp = msg.get('timestamp', '')
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime(' [%Y-%m-%d %H:%M:%S]')
            except:
                timestamp = ''
        
        print(f"{role}{timestamp}:")
        print(f"{content}")
        print("-" * 80)

def export_session(manager: ChatSessionManager, session_id: str, output_format: str, output_file: str = None):
    """Export a chat session to different formats"""
    session_path = manager.base_path / session_id
    
    if not session_path.exists():
        print(f"Error: Session '{session_id}' not found.")
        return
    
    conversation_file = session_path / "conversation.json"
    
    if not conversation_file.exists():
        print("No conversation file found in this session.")
        return
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if output_file is None:
        output_file = f"{session_id}.{output_format}"
    
    if output_format == 'json':
        # Pretty print JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif output_format == 'txt':
        # Export as plain text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Chat Session: {data.get('session_id', 'Unknown')}\n")
            f.write(f"Created: {data.get('created_at', 'Unknown')}\n")
            f.write(f"Last Updated: {data.get('last_updated', 'Unknown')}\n")
            f.write("=" * 80 + "\n\n")
            
            for msg in data.get('messages', []):
                role = msg['role'].upper()
                content = msg['content']
                timestamp = msg.get('timestamp', '')
                
                if timestamp:
                    f.write(f"\n[{timestamp}] ")
                f.write(f"{role}:\n{content}\n")
                f.write("-" * 80 + "\n")
    
    elif output_format == 'md':
        # Export as Markdown
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chat Session: {data.get('session_id', 'Unknown')}\n\n")
            f.write(f"**Created:** {data.get('created_at', 'Unknown')}  \n")
            f.write(f"**Last Updated:** {data.get('last_updated', 'Unknown')}\n\n")
            f.write("---\n\n")
            
            for msg in data.get('messages', []):
                role = msg['role']
                content = msg['content']
                timestamp = msg.get('timestamp', '')
                
                if role == 'system':
                    f.write(f"### System Prompt\n\n{content}\n\n")
                elif role == 'user':
                    f.write(f"### User")
                    if timestamp:
                        f.write(f" _{timestamp}_")
                    f.write(f"\n\n{content}\n\n")
                elif role == 'assistant':
                    f.write(f"### Assistant")
                    if timestamp:
                        f.write(f" _{timestamp}_")
                    f.write(f"\n\n{content}\n\n")
                
                f.write("---\n\n")
    
    print(f"Session exported to: {output_file}")

def clean_sessions(manager: ChatSessionManager, days: int):
    """Clean up sessions older than specified days"""
    print(f"\nCleaning up sessions older than {days} days...")
    manager.clear_old_sessions(days)
    print("Cleanup complete.")

def main():
    parser = argparse.ArgumentParser(description='Manage OpenRouter chat sessions')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all chat sessions')
    list_parser.add_argument('-l', '--limit', type=int, help='Limit number of sessions to display')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View a specific chat session')
    view_parser.add_argument('session_id', help='Session ID to view')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a chat session')
    export_parser.add_argument('session_id', help='Session ID to export')
    export_parser.add_argument('-f', '--format', choices=['json', 'txt', 'md'], 
                              default='txt', help='Export format (default: txt)')
    export_parser.add_argument('-o', '--output', help='Output filename')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up old sessions')
    clean_parser.add_argument('-d', '--days', type=int, default=30, 
                             help='Remove sessions older than this many days (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize chat manager
    manager = ChatSessionManager()
    
    if args.command == 'list':
        list_sessions(manager, args.limit)
    elif args.command == 'view':
        view_session(manager, args.session_id)
    elif args.command == 'export':
        export_session(manager, args.session_id, args.format, args.output)
    elif args.command == 'clean':
        clean_sessions(manager, args.days)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()