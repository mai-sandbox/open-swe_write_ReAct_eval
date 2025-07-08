# open-swe_write_ReAct_eval

A basic AI assistant built with LangGraph that can have conversations and perform simple tasks using tools.

## Overview

This is a conversational agent that demonstrates core LangGraph concepts including state management, tool calling, and basic workflow orchestration. The agent can engage in conversations and use tools when needed to help users.

## Key Features

### 💬 Conversational Interface
- Natural conversation flow with memory
- Maintains context within a conversation session
- Responds appropriately to user queries

### 🛠️ Basic Tool Usage
- **Web Search**: Can search the internet for current information
- **Calculator**: Performs basic mathematical calculations
- Smart tool selection based on user needs

### 🔄 Simple Workflow
- Determines when tools are needed vs. direct conversation
- Routes between conversation and tool usage appropriately
- Provides clear responses based on tool results

## Architecture

The agent uses LangGraph's core features:

- **State Management**: Tracks conversation messages and context
- **Tool Integration**: Two simple but useful tools
- **Routing Logic**: Decides between direct response and tool usage
- **Memory**: Maintains conversation history

## Capabilities

The agent can:

1. **Have Conversations**: Engage in natural dialog about various topics
2. **Search Information**: Find current information when users ask about recent events or facts
3. **Do Math**: Perform calculations when users ask mathematical questions
4. **Choose Appropriately**: Know when to use tools vs. respond directly
5. **Remember Context**: Reference earlier parts of the conversation

The agent should be able to converse with the user, search for weather using the weather tool, and use the calculator tool for arithmetic.
