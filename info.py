import googlesearch
import wikipedia

class Info():
    def __init__():
        """
        A module that helps to pull information from the internet
        """
        pass
    
    def find_wiki_page(search_term):
        # Search wiki for page
        results = wikipedia.search(search_term)
    
        if not results:
            print(f"No results found for {search_term}")

        for index in range(len(results)):
            try:
                # Get the summary
                return wikipedia.summary(results[index])
            except:
                pass
    