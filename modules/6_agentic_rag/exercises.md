# Module 6 Exercises: Agentic RAG

Test your understanding of agentic RAG concepts with these hands-on exercises.

---

## Exercise 1: Basic Agent Interaction ⭐

**Goal:** Run the demo and observe agent behavior

**Tasks:**
1. Run the demo: `python demo.py`
2. Watch the agent's reasoning process (verbose output)
3. Answer these questions:
   - Which tool did the agent select for each query?
   - How many reasoning steps did each query take?
   - Did the agent ever use multiple tools for one query?

**Expected Outcome:** Understanding of how agents select and use tools

---

## Exercise 2: Add a New Tool ⭐⭐

**Goal:** Create and integrate a new tool

**Task:** Add a `SearchByPriority` tool that filters tickets by priority level.

**Steps:**
1. Open `tools.py`
2. Add a new method `search_by_priority(self, priority: str) -> str`
3. Add the tool to the `get_tools()` method
4. Test with queries like "Show me all critical priority tickets"

**Hints:**
```python
def search_by_priority(self, priority: str) -> str:
    """Find tickets by priority level (Critical, High, Medium, Low)"""
    # Your code here
    pass
```

**Expected Outcome:** A working tool that filters by priority

---

## Exercise 3: Improve Tool Descriptions ⭐⭐

**Goal:** Make the agent choose tools more accurately

**Current Problem:** The agent sometimes selects the wrong tool for certain queries.

**Task:** Improve the tool descriptions in `tools.py` to be more specific.

**Test Queries:**
- "What database problems have we seen?" (Should use SearchByCategory)
- "Show ticket TICK-010" (Should use GetTicketByID)
- "How many tickets do we have?" (Should use GetTicketStatistics)

**Questions:**
- What keywords help the agent choose correctly?
- How detailed should descriptions be?

**Expected Outcome:** More accurate tool selection

---

## Exercise 4: Custom Agent Prompt ⭐⭐

**Goal:** Modify agent behavior through prompt engineering

**Task:** Modify the system prompt in `demo.py` to make the agent:
1. Always mention which tool it used in the final answer
2. Suggest related tickets when answering
3. Ask clarifying questions when the query is ambiguous

**Test Query:** "Issues with users logging in"

**Expected Outcome:** Agent behavior changes based on your prompt

---

## Exercise 5: Conversation History ⭐⭐⭐

**Goal:** Build a multi-turn conversation agent

**Task:** Create a script that allows interactive conversation with the agent.

**Requirements:**
- Accept user input in a loop
- Maintain conversation history
- Allow exit with "quit" or "exit"
- Display agent's tool usage

**Starter Code:**
```python
from demo import conversational_agent

print("Support Assistant (type 'quit' to exit)")
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        break
    
    # Your code here
    pass
```

**Expected Outcome:** Interactive chat with the agent

---

## Exercise 6: Multi-Step Query ⭐⭐⭐

**Goal:** Test agent's multi-step reasoning

**Task:** Write queries that require the agent to use multiple tools.

**Example Queries:**
1. "How many payment tickets do we have and what was the most recent one?"
2. "Find all high priority tickets and tell me which categories they fall into"
3. "Compare the resolution for TICK-001 and TICK-005"

**Questions:**
- Does the agent decompose the query correctly?
- What order does it use tools?
- Does it synthesize information well?

**Expected Outcome:** Understanding of multi-step reasoning

---

## Exercise 7: Add Ticket Creation Tool ⭐⭐⭐

**Goal:** Create a tool that modifies data (not just reads)

**Task:** Add a tool that creates a new support ticket.

**Requirements:**
- Accept: title, description, category, priority
- Generate a new ticket ID
- Add to the tickets list (in memory or file)
- Return confirmation with the ticket ID

**Hint:**
```python
def create_ticket(self, ticket_info: str) -> str:
    """
    Create a new support ticket.
    Input should be in format: 
    'title: [title], description: [desc], category: [cat], priority: [pri]'
    """
    # Parse the input
    # Create ticket
    # Add to list
    # Return confirmation
    pass
```

**Test Query:** "Create a new ticket: title: Test Issue, description: Testing agent, category: Other, priority: Low"

**Expected Outcome:** Functioning ticket creation

---

## Exercise 8: Error Handling ⭐⭐⭐

**Goal:** Make tools more robust

**Task:** Add error handling to tools for edge cases.

**Test Cases:**
1. Invalid ticket ID: "Get ticket TICK-999"
2. Invalid category: "Show me XYZ category tickets"
3. Malformed input: "Search for ;;;"

**Requirements:**
- Tools should never crash
- Return helpful error messages
- Suggest corrections when possible

**Expected Outcome:** Graceful error handling

---

## Exercise 9: Streaming Agent Responses ⭐⭐⭐⭐

**Goal:** Show agent thinking in real-time

**Task:** Implement streaming for better UX.

**Requirements:**
- Stream agent's intermediate steps
- Show tool calls as they happen
- Display final answer progressively

**Hint:** Look into `agent_executor.stream()` or `astream_events()`

**Expected Outcome:** Real-time visibility into agent process

---

## Exercise 10: Agent Evaluation ⭐⭐⭐⭐

**Goal:** Measure agent performance

**Task:** Create an evaluation script that tests the agent on a set of queries.

**Metrics to Track:**
1. Tool selection accuracy (did it choose the right tool?)
2. Number of steps taken
3. Response quality (manual or LLM-based grading)
4. Token usage
5. Latency

**Test Set:**
```python
test_queries = [
    ("How do I fix login issues?", "SearchSimilarTickets"),
    ("Show ticket TICK-001", "GetTicketByID"),
    ("How many tickets do we have?", "GetTicketStatistics"),
    # Add more...
]
```

**Expected Outcome:** Quantitative agent performance data

---

## Bonus Exercise: Hybrid RAG Router ⭐⭐⭐⭐⭐

**Goal:** Build a system that routes between direct RAG and agent

**Task:** Create a router that decides:
- Simple queries → Direct RAG (Module 4)
- Complex queries → Agent (Module 6)

**Classification Criteria:**
- Query complexity (word count, question marks)
- Intent (retrieval vs action)
- Multi-step indicators

**Architecture:**
```
User Query
    ↓
[Query Classifier]
    ↓
  ┌─────┴─────┐
  ↓           ↓
Direct RAG   Agent
  ↓           ↓
  └─────┬─────┘
        ↓
    Response
```

**Expected Outcome:** Efficient hybrid system

---

## Challenge Exercise: Multi-Agent System ⭐⭐⭐⭐⭐

**Goal:** Build multiple specialized agents

**Task:** Create a system with:
1. **Retrieval Agent** - Finds relevant tickets
2. **Analysis Agent** - Analyzes patterns and trends
3. **Response Agent** - Generates final user-facing answer
4. **Coordinator Agent** - Routes to appropriate agent

**Framework Options:**
- LangGraph for orchestration
- CrewAI for multi-agent collaboration
- Custom implementation

**Expected Outcome:** Working multi-agent system

---

## Reflection Questions

After completing exercises, consider:

1. **When would you use agentic RAG over direct RAG?**
2. **What are the cost implications of using agents?**
3. **How can you make tool selection more accurate?**
4. **What are the limits of current agents?**
5. **How would you deploy an agent in production?**

---

## Solutions

Solutions and example implementations available in `solutions/` directory (create your own first!).

---

## Next Steps

1. Explore LangGraph for more complex workflows
2. Try different agent types (ReAct, Plan-and-Execute)
3. Add external API tools (web search, databases)
4. Implement guardrails and safety checks
5. Build a full-stack application with the agent

Good luck! 🚀
