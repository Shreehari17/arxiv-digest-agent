from src.graph import build_graph
from src.state import initial_state


def main():
    user_input = input("Enter a research topic or arXiv ID: ").strip()

    if not user_input:
        print("Please enter something.")
        return

    graph = build_graph()
    state = initial_state(user_input)

    final_state = graph.invoke(state)

    if final_state.get("error"):
        print(f"\nError: {final_state['error']}")


if __name__ == "__main__":
    main()