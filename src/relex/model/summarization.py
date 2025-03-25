import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def summarize_results(results):
    # Summary of results
    print("\n=== Summary of Results ===")
    print("\nForecast Accuracy:")
    for item, result in results.items():
        print(
            f"{item}: RMSE = {result['metrics']['rmse']:.2f}, MAPE = {result['metrics']['mape']:.2f}%"
        )

    print("\nOwn-Price Elasticities:")
    for item_num in range(1, 5):
        own_idx = item_num - 1
        print(
            f"Item {item_num}: {results[f'item_{item_num}']['elasticities'][own_idx]:.4f}"
        )

    # Plot elasticities heatmap
    elasticity_matrix = np.zeros((4, 4))
    for i in range(4):
        elasticity_matrix[i] = results[f"item_{i + 1}"]["elasticities"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        elasticity_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=[f"Price {i + 1}" for i in range(4)],
        yticklabels=[f"Sales {i + 1}" for i in range(4)],
    )
    plt.title("Price Elasticity Matrix")
    plt.tight_layout()
    plt.show()

    # Identify major findings and implications
    print("\n=== Major Findings and Implications ===")

    # Check for negative own-price elasticities
    for item_num in range(1, 5):
        own_idx = item_num - 1
        elasticity = results[f"item_{item_num}"]["elasticities"][own_idx]
        if elasticity < 0:
            print(
                f"- Item {item_num} has expected negative own-price elasticity ({elasticity:.4f})"
            )
        else:
            print(
                f"- WARNING: Item {item_num} has unexpected positive own-price elasticity ({elasticity:.4f})"
            )

    # Check for strong substitution or complementary effects
    for i, j in itertools.product(range(1, 5), range(1, 5)):
        if i != j:
            cross_elasticity = results[f"item_{i}"]["elasticities"][j - 1]
            if cross_elasticity > 0.5:
                print(
                    f"- Items {j} and {i} appear to be substitutes (elasticity: {cross_elasticity:.4f})"
                )
            elif cross_elasticity < -0.5:
                print(
                    f"- Items {j} and {i} appear to be complements (elasticity: {cross_elasticity:.4f})"
                )

    # Check which components explain most of the variance
    for item_num in range(1, 5):
        component_means = results[f"item_{item_num}"]["component_means"]
        # total_variance = np.var(data["full_sales"])
        component_variances = {
            name: np.var(mean) for name, mean in component_means.items()
        }

        # Find the component with highest variance
        top_component = max(component_variances.items(), key=lambda x: x[1])
        print(
            f"- For Item {item_num}, the {top_component[0]} component explains most of the variance"
        )
