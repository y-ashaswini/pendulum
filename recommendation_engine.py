from IPython.display import clear_output
import numpy as np
import google.generativeai as genai

# Realtime Adaptive SLM
# - on the go relearning
# - can adapt to current behaviour and predict (extrapolate)
# - real time data ingestion
# - powerful lightweight recommendation engine

genai.configure(api_key="")

def get_embeddings(text):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )

    return np.array(result['embedding'])

class RecommendationEngine:
    """
    Recommender Engine class
    - create embedding vectors for items and init user vectors to 0
    - calculate a match score (dot prod) between a user and an item 
    - learning: if a user interacts with an item, nudge their vectors closer together using a small learning rate
    """

    def __init__(self, items, lr):
        self.items = items
        self.lr = lr
        
        # should be able to use matrix factorisation on it
        self.item_vectors = {
            iid: get_embeddings(category)
            for iid, category in items.items()
        }
        self.latent_dims = 3072 # gemini's output size
        self.user_vector = np.zeros(self.latent_dims) # will learn user preferences, start blank
      
    """Dot prod of user and current item id"""
    def predict_score(self, item_id):
        return np.dot(self.user_vector, self.item_vectors[item_id])

    """Online learning, reward=1 (clicked) reward=0 (not clicked)"""
    def update(self, item_id, reward):
        curr_item_vec = self.item_vectors[item_id]
        curr_pred_score = self.predict_score(item_id)
        error = reward - curr_pred_score
        
        # update user vector
        # move closer to item vector if high reward
        # FORMULA: vector_1 = vector_1 +(lr * error * vector_2)
        self.user_vector += self.lr * error * curr_item_vec

        # update item vector
        # makes system more adaptive to trends
        self.item_vectors[item_id] += self.lr * error * self.user_vector
        return abs(error)
    
    """Cadidate generaton, reranking all the items in the bank"""
    def get_top_n(self, n=5):
        scores = [(iid, self.predict_score(iid)) for iid in self.items.keys()]
        
        # sort by highest predicted score
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]

    """
        realtime data ingestion, gets embedding for the new item and adds to the live bank
    """
    def ingest_new_item(self):
        print(f"\n[new item detected]")

def run(engine):
    last_error = 0.0
    status_msg = "Start choosing to train the engine!"
    col_width = 30
    
    while True:
        current_recs = engine.get_top_n(10)

        clear_output(wait=True)
        print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print("┃            REALTIME ADAPTIVE ENGINE                ┃")
        print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        print(f"  [status]: {status_msg}")
        print(f"  [loss]: {last_error:.6f}")
        print(f"  [user vec (top 3)]: {engine.user_vector[:3]}")
        print("━" * 54)

        print("  TOP RECOMMENDATIONS (realtime):")
        for i, (iid, score) in enumerate(current_recs):
            # visualise score using bar
            bar = "█" * int(max(0, score * 10))
            print(f"{f'{i+1}. ({items[iid]})':<{col_width}} {iid:<{col_width}} {f'[{score:.2f}] {bar}':<{col_width}}")
            
        print("━" * 54)
        print("  [Actions]: 1-10 to Click | 0 to Exit")

        try:
            choice = input("\n  choose: ").strip()
            
            if choice == '0': 
                print("goodbye...")
                break
                
            idx = int(choice) - 1
            if 0 <= idx < len(current_recs):
                clicked_id = current_recs[idx][0]
                
                # clicked: reward=1
                last_error = engine.update(clicked_id, reward=1.0)
                
                # not clicked: reward=0
                for i, (iid, _) in enumerate(current_recs):
                    if i != idx:
                        engine.update(iid, reward=0.0)
                
                status_msg = f"user's choice: {clicked_id}"
            else:
                status_msg = "invalid choice. choose 1-10."
                
        except ValueError:
            status_msg = "invalid input."


items = {
    "Book1": "Tech,Home", "Book3": "Home", "book6": "Fitness",
    "Book5": "Tech", "Book2": "Home", "Book7": "Fitness,Home",
    "Book2": "Tech", "Book9": "Cooking,Tech", "Book11": "Cooking,Home",
    "Book8": "Tech", "Book10": "Cooking,Fitness", "Book12": "Tech"
}


engine = RecommendationEngine(items, lr=0.2)
run(engine)
