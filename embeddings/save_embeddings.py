import os
import pickle


def save_embedding(person_name, embeddings, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    output_file = os.path.join(embeddings_dir, f"{person_name}_embeddings.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings for {person_name} saved at {output_file}")
