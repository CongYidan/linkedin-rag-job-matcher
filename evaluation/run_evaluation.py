from retrieval.retrieval import JobRetriever
from evaluate import run_complete_evaluation

# Define test queries (this would be your test set)
test_queries = [
    {"title": "Software Engineer", "location": "San Francisco", "skills": "python,java,react"},
    {"title": "Data Scientist", "location": "New York", "skills": "python,machine learning,statistics"},
    {"title": "UX Designer", "location": "Boston", "skills": "figma,user research,prototyping"},
    {"title": "Product Manager", "location": "Seattle", "skills": "agile,product development,roadmap"},
    {"title": "DevOps Engineer", "location": "Remote", "skills": "aws,kubernetes,docker"},
]

# For a simple course project, you can use simulated ground truth
# In a real project, you would manually judge relevance
ground_truth = {
    0: ["12345", "23456", "34567"],  # Relevant job IDs for first query
    1: ["45678", "56789", "67890"],  # Relevant job IDs for second query
    # Add more as needed
}

# Initialize your retriever
retriever = JobRetriever()

# Run complete evaluation
evaluation_results = run_complete_evaluation(retriever, test_queries)

# Print key metrics
print("\nRETRIEVAL METRICS:")
for metric, value in evaluation_results["retrieval_metrics"].items():
    print(f"{metric}: {value:.3f}")

print("\nCOMPONENT IMPORTANCE:")
for component, value in evaluation_results["component_scores"].items():
    print(f"{component}: {value:.3f}")

print("\nVISUALIZATIONS SAVED TO:")
for name, path in evaluation_results["visualization_paths"].items():
    print(f"{name}: {path}")