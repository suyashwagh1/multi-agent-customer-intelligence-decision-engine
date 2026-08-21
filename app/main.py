import uuid

from langchain_core.messages import HumanMessage

from app.graph.build_graph import build_graph


def main():
    graph, checkpointer_cm = build_graph()
    config = {"configurable": {"thread_id": f"cli-test-{uuid.uuid4().hex[:8]}"}}

    print("OrderAgent ready. Type a message (Ctrl+C to quit).")
    print("Try: 'Where is my order ORD-000001?' (use a real order_id from your orders.csv)\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            for event in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
            ):
                for value in event.values():
                    messages = value.get("messages")
                    if not messages:
                        continue
                    message = messages[-1]
                    if message.content and isinstance(message.content, str):
                        print(f"Agent: {message.content}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        checkpointer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()