"""Find books similar to a title using a content-based k-nearest-neighbors model."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "books.csv"


def normalize_column_name(name: str) -> str:
    """Normalize common CSV column names such as Book-Title or book_title."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def value_from(row: dict[str, str | None], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class Book:
    title: str
    author: str
    genre: str
    description: str

    @property
    def searchable_text(self) -> str:
        # Repeat genre to give it useful weight alongside each book's description.
        return " ".join(
            part for part in (
                f"title {self.title}" if self.title else "",
                f"author {self.author}" if self.author else "",
                f"genre {self.genre} {self.genre}" if self.genre else "",
                self.description,
            ) if part
        )


class BookRecommender:
    """TF-IDF represents book metadata as vectors; k-NN finds nearby vectors."""

    def __init__(self, books: list[Book]) -> None:
        if len(books) < 2:
            raise ValueError("The CSV must contain at least two valid books.")
        self.books = books
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(book.searchable_text for book in books)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(matrix)
        self.matrix = matrix

    @classmethod
    def from_csv(cls, path: Path) -> "BookRecommender":
        if not path.is_file():
            raise FileNotFoundError(f"Could not find the book CSV: {path}")

        books: list[Book] = []
        seen_titles: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("The CSV is empty or does not have a header row.")
            for raw_row in reader:
                row = {
                    normalize_column_name(key): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                title = value_from(row, "title", "book_title", "name")
                if not title or title.casefold() in seen_titles:
                    continue
                book = Book(
                    title=title,
                    author=value_from(row, "author", "book_author", "authors"),
                    genre=value_from(row, "genre", "genres", "category", "categories", "subject"),
                    description=value_from(
                        row, "description", "summary", "synopsis", "about", "abstract"
                    ),
                )
                books.append(book)
                seen_titles.add(title.casefold())

        if len(books) < 2:
            raise ValueError(
                "The CSV needs at least two rows with a title column. "
                "Supported title headers include title and Book-Title."
            )
        return cls(books)

    def find_matches(self, query: str) -> list[int]:
        cleaned = query.strip().casefold()
        exact = [i for i, book in enumerate(self.books) if book.title.casefold() == cleaned]
        if exact:
            return exact
        return [i for i, book in enumerate(self.books) if cleaned in book.title.casefold()]

    def recommend(self, book_index: int, count: int = 5) -> list[tuple[Book, float]]:
        if count < 1:
            return []
        neighbors = min(count + 1, len(self.books))
        distances, indices = self.model.kneighbors(
            self.matrix.getrow(book_index), n_neighbors=neighbors
        )
        recommendations: list[tuple[Book, float]] = []
        for distance, index in zip(distances[0], indices[0]):
            if index == book_index:
                continue
            similarity = max(0.0, 1.0 - float(distance)) * 100
            recommendations.append((self.books[index], similarity))
            if len(recommendations) == count:
                break
        return recommendations


def print_recommendations(recommender: BookRecommender, book_index: int, count: int) -> None:
    book = recommender.books[book_index]
    print(f"\nBecause you selected: {book.title} by {book.author or 'Unknown author'}")
    recommendations = recommender.recommend(book_index, count)
    if not recommendations:
        print("No other books are available to recommend.")
        return
    for number, (suggestion, similarity) in enumerate(recommendations, start=1):
        author = f" by {suggestion.author}" if suggestion.author else ""
        print(f"{number}. {suggestion.title}{author}  (similarity: {similarity:.1f}%)")


def print_book_list(recommender: BookRecommender) -> None:
    print("\nBooks in the dataset:")
    for book in recommender.books:
        author = f" by {book.author}" if book.author else ""
        print(f"- {book.title}{author}")


def choose_title(recommender: BookRecommender, query: str) -> int | None:
    matches = recommender.find_matches(query)
    if not matches:
        print(f"No book title matched '{query}'. Try 'list' to see available titles.")
        return None
    if len(matches) == 1:
        return matches[0]
    print("More than one title matched. Choose one:")
    for number, index in enumerate(matches, start=1):
        book = recommender.books[index]
        author = f" by {book.author}" if book.author else ""
        print(f"{number}. {book.title}{author}")
    while True:
        selection = input("Enter a number (or press Enter to cancel): ").strip()
        if not selection:
            return None
        if selection.isdigit() and 1 <= int(selection) <= len(matches):
            return matches[int(selection) - 1]
        print("Please enter one of the listed numbers.")


def run_interactive(recommender: BookRecommender, count: int) -> None:
    print("\nBook Similarity Recommender (k-NN)")
    print(f"Loaded {len(recommender.books)} books. Type 'list' to browse or 'quit' to exit.")
    while True:
        try:
            query = input("\nEnter a book title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if query.casefold() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return
        if query.casefold() in {"list", "books"}:
            print_book_list(recommender)
            continue
        if not query:
            continue
        index = choose_title(recommender, query)
        if index is not None:
            print_recommendations(recommender, index, count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recommend books with a content-based k-nearest-neighbors model."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to a books CSV file")
    parser.add_argument("--title", help="Recommend books similar to this title, then exit")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to show (default: 5)")
    parser.add_argument("--list", action="store_true", help="Print all titles in the dataset")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")
    try:
        recommender = BookRecommender.from_csv(args.data)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print_book_list(recommender)
    elif args.title:
        index = choose_title(recommender, args.title)
        if index is None:
            return 1
        print_recommendations(recommender, index, args.top)
    else:
        run_interactive(recommender, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
