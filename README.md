# Similar Book Recommendation with k-NN

This AI project recommends books that are similar to a title you choose. It uses a content-based **k-nearest-neighbors (k-NN)** model: TF-IDF turns each book's title, author, genre, and description into a numeric vector, then cosine distance finds the closest book vectors.

The included dataset is a small example so the project runs immediately. You can use your own CSV with the same kind of book information.

## Set up

From this folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The book recommender uses scikit-learn. The OpenAI package remains in the requirements for the separate `Jarvis.py` assistant.

## Run

Start interactive mode:

```powershell
python main.py
```

Enter a book title, type `list` to browse the dataset, or type `quit` to exit. You can also request recommendations directly:

```powershell
python main.py --title "The Hobbit"
python main.py --title "Pride and Prejudice" --top 3
python main.py --list
```

Use another CSV by passing its path:

```powershell
python main.py --data "path\to\my_books.csv" --title "Dune"
```

## Your CSV file

The CSV must have a header row and at least two book rows. The title column is required. Author, genre, and description make recommendations more useful; the program also recognizes common alternatives such as `Book-Title`, `Book-Author`, `categories`, and `summary`.

```csv
title,author,genre,description
The Hobbit,J.R.R. Tolkien,Fantasy,"A hobbit joins a quest to reclaim treasure from a dragon."
Dune,Frank Herbert,Science Fiction,"A young noble becomes involved in politics on a desert planet."
```

## How the model works

1. The program combines each book's title, author, genre, and description into text.
2. TF-IDF gives more weight to informative words and less weight to common words.
3. k-NN finds the `k` closest books using cosine distance. The closest book is shown first.

The displayed similarity percentage is `1 - cosine distance`; it is a comparison score, not a probability that you will like the book. This is a content-based recommender, so it does not need user ratings or an internet connection.
