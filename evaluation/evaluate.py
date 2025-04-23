import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
import json

def evaluate_retrieval_performance(retriever, test_queries, k=5):
    """
    Evaluate retrieval metrics including precision, recall, F1, and NDCG.
    
    Args:
        retriever: JobRetriever instance
        test_queries: List of test queries with filters
        k: Number of top results to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        "avg_top1_similarity": [],
        "avg_top3_similarity": [],
        "avg_top5_similarity": [],
        "similarity_dropoff": [],  # Measure of how quickly similarity drops
        "component_consistency": []  # Consistency of component scores across results
    }
    
    for query in test_queries:
        search_results = retriever.search_jobs(
            query=query["title"],
            filters={
                "location": query.get("location"),
                "skills": query.get("skills"),
                "remote": query.get("remote", False)
            },
            n_results=k
        )
        
        if not search_results:
            continue
            
        # Get similarity scores
        similarities = [result["similarity"] for result in search_results]
        
        # Calculate metrics
        results["avg_top1_similarity"].append(similarities[0] if similarities else 0)
        results["avg_top3_similarity"].append(np.mean(similarities[:3]) if len(similarities) >= 3 else np.mean(similarities))
        results["avg_top5_similarity"].append(np.mean(similarities) if similarities else 0)
        
        # Calculate dropoff (difference between top and average)
        if similarities and len(similarities) > 1:
            results["similarity_dropoff"].append(similarities[0] - np.mean(similarities[1:]))
        
        # Calculate component consistency (standard deviation of component scores)
        if len(search_results) > 1 and "component_scores" in search_results[0]:
            components = search_results[0]["component_scores"].keys()
            component_stds = {}
            
            for component in components:
                scores = [result["component_scores"][component] for result in search_results 
                         if "component_scores" in result]
                if scores:
                    component_stds[component] = np.std(scores)
            
            # Average consistency across components (lower std = more consistent)
            avg_std = np.mean(list(component_stds.values())) if component_stds else 0
            results["component_consistency"].append(1 - min(avg_std, 1))  # Convert to 0-1 scale, higher is better
        
    # Calculate averages
    avg_results = {metric: np.mean(values) for metric, values in results.items() if values}
    return avg_results

def evaluate_components(retriever, test_queries):
    """
    Evaluate the performance of individual similarity components.
    
    Args:
        retriever: JobRetriever instance
        test_queries: List of test queries with filters
    
    Returns:
        Dictionary with average component scores
    """
    component_scores = {
        "semantic": [],
        "title": [],
        "skills": [],
        "location": []
    }
    
    for query in test_queries:
        results = retriever.search_jobs(
            query=query["title"],
            filters={
                "location": query.get("location"),
                "skills": query.get("skills"),
                "remote": query.get("remote", False)
            },
            n_results=5
        )
        
        # Extract component scores from results
        for result in results:
            if "component_scores" in result:
                for component, score in result["component_scores"].items():
                    if component in component_scores:
                        component_scores[component].append(score)
    
    # Calculate averages
    avg_components = {component: np.mean(scores) for component, scores in component_scores.items() if scores}
    return avg_components

def perform_end_to_end_evaluation(retriever, test_queries):
    """
    Perform end-to-end evaluation with sample queries and analyze results.
    
    Args:
        retriever: JobRetriever instance
        test_queries: List of test queries with title, location, and skills
    
    Returns:
        DataFrame with result data
    """
    results_data = []
    
    for query in test_queries:
        query_str = f"{query['title']}"
        if query.get('location'):
            query_str += f" in {query['location']}"
        if query.get('skills'):
            query_str += f" with skills: {query['skills']}"
        
        results = retriever.search_jobs(
            query=query["title"],
            filters={
                "location": query.get("location"),
                "skills": query.get("skills"),
                "remote": query.get("remote", False)
            },
            n_results=5
        )
        
        # Store results for visualization
        for i, job in enumerate(results):
            job_data = {
                "query": query_str,
                "rank": i+1,
                "title": job['title'],
                "company": job['company'],
                "similarity": job['similarity']
            }
            
            # Add component scores if available
            if "component_scores" in job:
                for component, score in job["component_scores"].items():
                    job_data[f"{component}_score"] = score
            
            results_data.append(job_data)
    
    return pd.DataFrame(results_data)

def plot_retrieval_metrics(metrics, save_path="evaluation/retrieval_metrics.png"):
    """Plot retrieval metrics as a bar chart."""
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Retrieval Quality Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_component_importance(component_scores, save_path="evaluation/component_importance.png"):
    """Plot component importance as a bar chart."""
    plt.figure(figsize=(10, 6))
    plt.bar(component_scores.keys(), component_scores.values(), color='lightgreen')
    plt.title("Component Contribution to Similarity")
    plt.xlabel("Component")
    plt.ylabel("Average Score")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_similarity_by_rank(results_df, save_path="evaluation/similarity_by_rank.png"):
    """Plot similarity scores by rank."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="rank", y="similarity", data=results_df, palette="Blues")
    plt.title("Similarity Score Distribution by Rank")
    plt.xlabel("Rank")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_query_performance(results_df, save_path="evaluation/query_performance.png"):
    """Plot query performance comparison."""
    plt.figure(figsize=(12, 8))
    sns.barplot(x="query", y="similarity", data=results_df, ci=None, palette="muted")
    plt.title("Average Similarity by Query")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def run_complete_evaluation(retriever, test_queries):
    """
    Run complete evaluation suite and generate visualizations.
    
    Args:
        retriever: JobRetriever instance
        test_queries: List of test queries
    
    Returns:
        Dictionary with evaluation results and paths to visualizations
    """
 
    # Perform evaluations
    print("Evaluating retrieval performance...")
    retrieval_metrics = evaluate_retrieval_performance(retriever, test_queries,)
    
    print("Evaluating component importance...")
    component_scores = evaluate_components(retriever, test_queries)
    
    print("Performing end-to-end evaluation...")
    results_df = perform_end_to_end_evaluation(retriever, test_queries)
    
    # Generate visualizations
    print("Generating visualizations...")
    viz_paths = {}
    viz_paths["retrieval_metrics"] = plot_retrieval_metrics(retrieval_metrics)
    viz_paths["component_importance"] = plot_component_importance(component_scores)
    viz_paths["similarity_by_rank"] = plot_similarity_by_rank(results_df)
    viz_paths["query_performance"] = plot_query_performance(results_df)
    
    # Save evaluation results as CSV
    results_df.to_csv("evaluation/end_to_end_results.csv", index=False)
    
    # Create summary results dictionary
    results_summary = {
        "retrieval_metrics": retrieval_metrics,
        "component_scores": component_scores,
        "visualization_paths": viz_paths
    }
    
    # Save summary results as JSON
    with open("evaluation/evaluation_summary.json", "w") as f:
        json.dump({k: v for k, v in results_summary.items() if k != "visualization_paths"}, f, indent=2)
    
    print("Evaluation complete. Results saved to evaluation/ directory.")
    return results_summary