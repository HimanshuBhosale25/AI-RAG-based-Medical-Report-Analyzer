from pubmed_search import search_pubmed
from pubmed_fetch import fetch_article_details

def main():
    # Ask the user for a search query
    query = input("Enter a search query for PubMed: ")
    
    # Step 1: Search PubMed
    print(f"Searching PubMed for: {query}...")
    pmids = search_pubmed(query)
    
    if pmids:
        print(f"Found {len(pmids)} articles. Fetching details...")
        
        # Step 2: Fetch article details
        articles = fetch_article_details(pmids)
        
        if articles:
            print(f"Fetched details for {len(articles)} articles:\n")
            
            # Display article details (title, source, pub date, abstract)
            for i, article in enumerate(articles, start=1):
                print(f"Article {i}:")
                print(f"Title: {article.get('title', 'N/A')}")
                print(f"Source: {article.get('source', 'N/A')}")
                print(f"Published Date: {article.get('pub_date', 'N/A')}")
                print(f"Abstract: {article.get('abstract', 'N/A')}\n")
        else:
            print("No article details found.")
    else:
        print("No articles found for the search query.")

if __name__ == "__main__":
    main()
