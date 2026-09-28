from src.graph import build_graph
from src.state import initial_state


def main():
    print("\n" + "=" * 60)
    print("              PAPER ID/URL OR TOPIC TO RESEARCH")
    print("=" * 60)

    user_input = input("❯ ").strip()

    if user_input.lower() in {"quit", "exit"}:
        print("\n" + "=" * 60)
        print("                       SESSION ENDED")
        print("=" * 60)
        return

    graph = build_graph()
    state = initial_state(user_input)

    try:
        final_state = graph.invoke(state)

        if final_state.get("error"):
            print(f"\nError: {final_state['error']}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("                       SESSION ENDED")
        print("=" * 60)


if __name__ == "__main__":
    main()